from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

from src.analysis.claude_analyst import ClaudeAnalyst
from src.backtest.baselines import RandomAnalyst, SmaCrossoverAnalyst, buy_and_hold_pnl
from src.backtest.cache import ResponseCache
from src.backtest.engine import (
    BacktestConfig,
    BacktestEngine,
    DryRunAnalyst,
    prompt_fingerprint,
)
from src.backtest.report import format_report
from src.risk.manager import RiskManager
from src.utils.logger import setup_logging


def load_bars(path: str, start: str = None, end: str = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize column names
    df.columns = [c.lower() for c in df.columns]
    if "date" in df.columns and "timestamp" not in df.columns:
        df = df.rename(columns={"date": "timestamp"})
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if start:
        df = df[df["timestamp"] >= pd.to_datetime(start)]
    if end:
        df = df[df["timestamp"] <= pd.to_datetime(end)]
    return df.sort_values("timestamp").reset_index(drop=True)


async def run():
    parser = argparse.ArgumentParser(description="Replay historical bars through the trading agent pipeline")
    parser.add_argument("--data", required=True, help="CSV file with OHLCV bars")
    parser.add_argument("--instrument", required=True, help="Instrument key (e.g. NQ, ES, XAUUSD)")
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--prompts-dir", default="config/prompts")
    parser.add_argument("--start", default=None, help="ISO date to start from")
    parser.add_argument("--end", default=None, help="ISO date to stop at")
    parser.add_argument("--multiplier", type=float, default=None, help="Contract multiplier (NQ=20, ES=50, XAUUSD=1)")
    parser.add_argument("--sample-every", type=int, default=1, help="Only analyze every Nth bar")
    parser.add_argument("--strategy", choices=["claude", "random", "sma", "dry"], default="claude",
                        help="Which analyst to run (claude=live LLM, random/sma=baselines, dry=HOLD stub)")
    parser.add_argument("--dry-run", action="store_true", help="Shortcut for --strategy dry")
    parser.add_argument("--no-cache", action="store_true", help="Disable response cache")
    parser.add_argument("--output", default=None, help="Path to write trade log JSON")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Override Claude temperature (recommend 0.0 for reproducibility)")
    args = parser.parse_args()

    load_dotenv()
    setup_logging(level="INFO")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    df = load_bars(args.data, args.start, args.end)
    if len(df) == 0:
        print("No bars in selected date range", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(df)} bars from {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Multiplier defaults by instrument
    default_mults = {
        "NQ": 20.0, "ES": 50.0, "XAUUSD": 1.0,
        "MNQ": 2.0, "MES": 5.0, "MGC": 10.0,
    }
    multiplier = args.multiplier if args.multiplier is not None else default_mults.get(args.instrument, 1.0)

    analysis_cfg = cfg["analysis"]
    ind_cfg = analysis_cfg["indicators"]
    bt_config = BacktestConfig(
        instrument=args.instrument,
        multiplier=multiplier,
        lookback_bars=analysis_cfg.get("lookback_bars", 50),
        bar_size=analysis_cfg["bar_size"],
        ema_periods=ind_cfg["ema_periods"],
        rsi_period=ind_cfg["rsi_period"],
        atr_period=ind_cfg["atr_period"],
        bb_period=ind_cfg["bb_period"],
        bb_std=ind_cfg["bb_std"],
        sample_every_n_bars=args.sample_every,
    )

    risk_cfg = cfg["risk"]
    risk_manager = RiskManager(
        max_position_size=risk_cfg["max_position_size"],
        max_concurrent_positions=risk_cfg["max_concurrent_positions"],
        daily_loss_limit_pct=risk_cfg["daily_loss_limit_pct"],
        min_confidence=risk_cfg["min_confidence"],
        risk_per_trade_pct=risk_cfg["risk_per_trade_pct"],
        account_balance=bt_config.starting_balance,
    )

    strategy = "dry" if args.dry_run else args.strategy
    if strategy == "dry":
        analyst = DryRunAnalyst()
    elif strategy == "random":
        analyst = RandomAnalyst()
    elif strategy == "sma":
        analyst = SmaCrossoverAnalyst()
    else:
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            print("ANTHROPIC_API_KEY not set (use --strategy dry/random/sma to skip)", file=sys.stderr)
            sys.exit(1)
        claude_cfg = cfg["claude"]
        temperature = args.temperature if args.temperature is not None else claude_cfg["temperature"]
        if temperature > 0:
            print(f"WARNING: temperature={temperature} — backtest results not reproducible. Use --temperature 0.0 for determinism.")
        analyst = ClaudeAnalyst(
            api_key=api_key,
            model=claude_cfg["model"],
            max_tokens=claude_cfg["max_tokens"],
            temperature=temperature,
            max_retries=claude_cfg["max_retries"],
        )

        # Warn if backtest window overlaps Claude's training data
        cutoff = pd.Timestamp("2025-08-01")
        if df["timestamp"].min() < cutoff:
            print(
                "WARNING: backtest includes data before 2025-08-01 (Claude Sonnet 4.6 training cutoff).\n"
                "         Results may be contaminated by training-set leakage.\n"
                "         Prefer out-of-sample data after the cutoff for honest measurement."
            )

    cache = None if args.no_cache else ResponseCache()
    fingerprint = prompt_fingerprint(args.prompts_dir)

    engine = BacktestEngine(
        df=df,
        config=bt_config,
        analyst=analyst,
        risk_manager=risk_manager,
        cache=cache,
        prompt_fingerprint=fingerprint,
    )

    stats = await engine.run()
    print(format_report(stats))

    bh_pnl = buy_and_hold_pnl(df, multiplier=multiplier)
    print(f"Buy-and-hold reference:  ${bh_pnl:,.2f}")
    if stats.total_pnl != 0 or bh_pnl != 0:
        delta = stats.total_pnl - bh_pnl
        verdict = "BEATS" if delta > 0 else "TRAILS"
        print(f"Strategy vs buy-and-hold: {verdict} by ${abs(delta):,.2f}")
    print(f"Strategy used:           {strategy}")

    if args.output:
        engine.write_trade_log(args.output)
        print(f"Trade log written to {args.output}")


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
