from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from src.analysis.claude_analyst import ClaudeAnalyst
from src.analysis.market_snapshot import build_market_snapshot
from src.backtest.cache import ResponseCache
from src.backtest.fill_sim import SimulatedTrade, simulate_bracket
from src.backtest.report import BacktestStats, compute_stats, format_report
from src.models.schemas import TradeAction, TradeDecision, TradeOutcome
from src.risk.manager import RiskManager
from src.utils.logger import get_logger


@dataclass
class BacktestConfig:
    instrument: str
    multiplier: float = 1.0
    lookback_bars: int = 50
    bar_size: str = "5 mins"
    ema_periods: list[int] = field(default_factory=lambda: [9, 21, 50])
    rsi_period: int = 14
    atr_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    starting_balance: float = 100_000.0
    sample_every_n_bars: int = 1


class DryRunAnalyst:
    """Stub analyst that always returns HOLD. Use to validate plumbing without API cost."""

    async def analyze(self, snapshot) -> TradeDecision:
        return TradeDecision(
            action=TradeAction.HOLD,
            instrument=snapshot.instrument,
            confidence=0.0,
            reasoning="dry-run stub",
        )


class BacktestEngine:
    def __init__(
        self,
        df: pd.DataFrame,
        config: BacktestConfig,
        analyst,
        risk_manager: RiskManager,
        cache: Optional[ResponseCache] = None,
        prompt_fingerprint: str = "default",
    ):
        self.df = df.sort_values("timestamp").reset_index(drop=True)
        self.config = config
        self.analyst = analyst
        self.risk_manager = risk_manager
        self.cache = cache
        self.prompt_fingerprint = prompt_fingerprint
        self.logger = get_logger()

        self.trades: list[SimulatedTrade] = []
        self.decisions_considered = 0
        self.decisions_approved = 0
        self.held_positions = 0  # trades skipped because already in position

    async def _decide(self, snapshot) -> Optional[TradeDecision]:
        if self.cache:
            cached = self.cache.get(snapshot, self.prompt_fingerprint)
            if cached is not None:
                return cached
        decision = await self.analyst.analyze(snapshot)
        if decision and self.cache:
            self.cache.put(snapshot, self.prompt_fingerprint, decision)
        return decision

    def _open_position_at(self, t: datetime) -> Optional[SimulatedTrade]:
        for trade in reversed(self.trades):
            if trade.entry_time <= t <= trade.exit_time:
                return trade
        return None

    async def run(self) -> BacktestStats:
        cfg = self.config
        n = len(self.df)
        self.logger.info(f"Backtest: {n} bars for {cfg.instrument}")

        # Need at least lookback_bars of history and 2 future bars to simulate fills
        for i in range(cfg.lookback_bars, n - 2):
            if (i - cfg.lookback_bars) % cfg.sample_every_n_bars != 0:
                continue

            history = self.df.iloc[: i + 1]
            current_bar = history.iloc[-1]
            current_time = current_bar["timestamp"]
            current_price = float(current_bar["close"])

            # Skip if already in a simulated position — live system would HOLD
            if self._open_position_at(current_time):
                self.held_positions += 1
                continue

            positions = []  # flat
            balance = cfg.starting_balance + sum(t.pnl for t in self.trades)

            # Sync daily pnl into risk manager from synthetic equity
            today_pnl = sum(
                t.pnl for t in self.trades
                if t.exit_time.date() == current_time.date()
            )
            self.risk_manager.sync_daily_pnl(today_pnl)
            self.risk_manager.update_balance(balance)
            self.risk_manager.update_positions(0)

            snapshot = build_market_snapshot(
                instrument=cfg.instrument,
                timeframe=cfg.bar_size,
                df=history,
                current_price=current_price,
                positions=positions,
                account_balance=balance,
                daily_pnl=today_pnl,
                ema_periods=cfg.ema_periods,
                rsi_period=cfg.rsi_period,
                atr_period=cfg.atr_period,
                bb_period=cfg.bb_period,
                bb_std=cfg.bb_std,
                recent_trades=self.risk_manager.recent_trades(),
            )

            decision = await self._decide(snapshot)
            if decision is None:
                continue

            self.decisions_considered += 1
            approved, reason = self.risk_manager.validate(decision)
            if not approved or decision.action == TradeAction.HOLD:
                continue

            self.decisions_approved += 1
            future = self.df.iloc[i + 1 :]
            trade = simulate_bracket(
                decision, future, multiplier=cfg.multiplier, decision_time=current_time
            )
            if trade is None:
                continue

            self.trades.append(trade)
            outcome = TradeOutcome(
                instrument=trade.instrument,
                action=trade.action.value,
                entry_time=trade.entry_time,
                entry_price=trade.entry_price,
                exit_time=trade.exit_time,
                exit_price=trade.exit_price,
                pnl=trade.pnl,
                exit_reason=trade.exit_reason,
                reasoning=trade.reasoning[:200],
            )
            self.risk_manager.record_trade_result(trade.pnl, outcome=outcome)
            self.logger.info(
                f"Trade: {trade.action.value} {cfg.instrument} "
                f"entry={trade.entry_price} exit={trade.exit_price} "
                f"pnl=${trade.pnl:.2f} reason={trade.exit_reason}"
            )

        self.logger.info(
            f"Done. Decisions: {self.decisions_considered} considered, "
            f"{self.decisions_approved} approved, {len(self.trades)} trades, "
            f"{self.held_positions} skipped (already in position)"
        )
        return compute_stats(self.trades)

    def write_trade_log(self, path: str):
        out = [
            {
                "instrument": t.instrument,
                "action": t.action.value,
                "entry_time": t.entry_time.isoformat() if hasattr(t.entry_time, "isoformat") else str(t.entry_time),
                "entry_price": t.entry_price,
                "exit_time": t.exit_time.isoformat() if hasattr(t.exit_time, "isoformat") else str(t.exit_time),
                "exit_price": t.exit_price,
                "size": t.size,
                "stop_loss": t.stop_loss,
                "take_profit": t.take_profit,
                "exit_reason": t.exit_reason,
                "pnl": t.pnl,
                "confidence": t.confidence,
                "reasoning": t.reasoning,
            }
            for t in self.trades
        ]
        Path(path).write_text(json.dumps(out, indent=2, default=str))


def prompt_fingerprint(prompts_dir: str) -> str:
    """Hash of all prompt markdown files — bump cache key when prompts change."""
    h = hashlib.sha256()
    for p in sorted(Path(prompts_dir).glob("*.md")):
        h.update(p.read_bytes())
    return h.hexdigest()[:16]
