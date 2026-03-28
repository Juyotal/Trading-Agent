from __future__ import annotations

import asyncio
import os
import signal
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.analysis.claude_analyst import ClaudeAnalyst
from src.analysis.market_snapshot import build_market_snapshot
from src.broker.connection import IBConnection
from src.broker.data_feed import DataFeed
from src.broker.executor import TradeExecutor
from src.models.schemas import SessionContext
from src.risk.manager import RiskManager
from src.utils.logger import setup_logging, get_logger
from src.utils.scheduler import MarketScheduler


class TradingAgent:
    """Main orchestrator: connects all components and runs the trading loop."""

    def __init__(self, config_path: str = "config/settings.yaml"):
        load_dotenv()
        self.config = self._load_config(config_path)
        self.logger = setup_logging(level="INFO")
        self._shutdown = False

        broker_cfg = self.config["broker"]
        self.connection = IBConnection(
            host=os.getenv("IB_HOST", broker_cfg["host"]),
            port=int(os.getenv("IB_PORT", broker_cfg["port"])),
            client_id=int(os.getenv("IB_CLIENT_ID", broker_cfg["client_id"])),
            timeout=broker_cfg.get("timeout", 30),
            auto_reconnect=broker_cfg.get("auto_reconnect", True),
        )
        self.data_feed = DataFeed(self.connection)

        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            self.logger.error("ANTHROPIC_API_KEY not set in environment")
            sys.exit(1)

        claude_cfg = self.config["claude"]
        self.analyst = ClaudeAnalyst(
            api_key=api_key,
            model=claude_cfg["model"],
            max_tokens=claude_cfg["max_tokens"],
            temperature=claude_cfg["temperature"],
            max_retries=claude_cfg["max_retries"],
        )

        risk_cfg = self.config["risk"]
        self.risk_manager = RiskManager(
            max_position_size=risk_cfg["max_position_size"],
            max_concurrent_positions=risk_cfg["max_concurrent_positions"],
            daily_loss_limit_pct=risk_cfg["daily_loss_limit_pct"],
            min_confidence=risk_cfg["min_confidence"],
            risk_per_trade_pct=risk_cfg["risk_per_trade_pct"],
            memory_path=risk_cfg.get("memory_path", "trade_memory.jsonl"),
            memory_size=risk_cfg.get("memory_size", 20),
        )
        self.executor = TradeExecutor(self.connection, self.risk_manager)

        sched_cfg = self.config["schedule"]
        self.scheduler = MarketScheduler(
            timezone=sched_cfg["timezone"],
            futures_start=sched_cfg["market_hours"]["futures_start"],
            futures_end=sched_cfg["market_hours"]["futures_end"],
        )

        self.interval = sched_cfg["interval_minutes"] * 60
        self.contracts = {}

        self.session_cfg = self.config.get("session", {})
        self._weekend_flattened = False

    @staticmethod
    def _load_config(path: str) -> dict:
        with open(path) as f:
            return yaml.safe_load(f)

    def _check_trading_mode(self):
        mode = os.getenv("TRADING_MODE", self.config.get("trading_mode", "paper"))
        if mode != "paper":
            self.logger.warning(
                "=" * 60 + "\n"
                "  WARNING: TRADING_MODE is set to LIVE\n"
                "  Real money is at risk. Proceed with caution.\n" +
                "=" * 60
            )
            confirm = input("Type 'CONFIRM LIVE' to proceed: ")
            if confirm.strip() != "CONFIRM LIVE":
                self.logger.info("Live trading not confirmed. Exiting.")
                sys.exit(0)
        else:
            self.logger.info("Running in PAPER trading mode")

    async def startup(self):
        self.logger.info("=" * 50)
        self.logger.info("  Trading Agent Starting Up")
        self.logger.info("=" * 50)

        self._check_trading_mode()

        connected = await self.connection.connect()
        if not connected:
            self.logger.error("Failed to connect to IB Gateway. Is it running?")
            sys.exit(1)

        summary = self.connection.get_account_summary()
        if summary:
            balance = summary.get("NetLiquidation", 0)
            self.risk_manager.update_balance(balance)
            self.logger.info(f"Account balance: ${balance:,.2f}")

        for name, inst_cfg in self.config["instruments"].items():
            contract = self.connection.make_contract(
                symbol=inst_cfg["symbol"],
                exchange=inst_cfg["exchange"],
                sec_type=inst_cfg["sec_type"],
                currency=inst_cfg.get("currency", "USD"),
            )
            qualified = await self.connection.qualify_contract(contract)
            if qualified:
                self.contracts[name] = qualified
                self.logger.info(f"Qualified contract: {name} -> {qualified}")
            else:
                self.logger.warning(f"Could not qualify contract for {name}")

        if not self.contracts:
            self.logger.error("No contracts qualified. Check instrument config.")
            sys.exit(1)

        session = self.scheduler.get_session_info()
        self.logger.info(f"Session info: {session}")

    def _build_session_context(self) -> SessionContext:
        close_buf = self.session_cfg.get("session_close_buffer_minutes", 15)
        weekend_buf = self.session_cfg.get("weekend_close_buffer_minutes", 30)
        first_bar_win = self.session_cfg.get("first_bar_window_minutes", 10)
        return SessionContext(
            first_bar_of_session=self.scheduler.is_first_bar_of_session(first_bar_win),
            near_session_close=self.scheduler.is_near_session_close(close_buf),
            near_weekend_close=self.scheduler.is_near_weekend_close(weekend_buf),
            minutes_until_close=self.scheduler.minutes_until_futures_close(),
        )

    def _should_block_new_entry(self, session_ctx: SessionContext, gap_anomalous: bool) -> tuple[bool, str]:
        if session_ctx.near_weekend_close:
            return True, "near weekend close — no new entries"
        if session_ctx.near_session_close:
            return True, f"near session close ({session_ctx.minutes_until_close}m) — no new entries"
        if session_ctx.first_bar_of_session and gap_anomalous:
            return True, "first bar of session with anomalous gap — no new entries"
        return False, ""

    async def analyze_instrument(self, name: str):
        contract = self.contracts[name]
        inst_cfg = self.config["instruments"][name]

        if not self.scheduler.is_market_open(inst_cfg["sec_type"]):
            self.logger.info(f"Market closed for {name} ({inst_cfg['sec_type']}), skipping")
            return

        analysis_cfg = self.config["analysis"]
        df = await self.data_feed.get_historical_bars(
            contract,
            bar_size=analysis_cfg["bar_size"],
            duration="1 D",
        )

        if df is None or len(df) < 20:
            self.logger.warning(f"Insufficient data for {name}")
            return

        current_price = df["close"].iloc[-1]
        positions = await self.data_feed.get_positions()
        summary = self.connection.get_account_summary()
        balance = summary.get("NetLiquidation", 100_000)
        daily_pnl = summary.get("RealizedPnL", 0)

        indicator_cfg = analysis_cfg["indicators"]
        session_ctx = self._build_session_context()
        snapshot = build_market_snapshot(
            instrument=name,
            timeframe=analysis_cfg["bar_size"],
            df=df,
            current_price=float(current_price),
            positions=positions,
            account_balance=balance,
            daily_pnl=daily_pnl,
            ema_periods=indicator_cfg["ema_periods"],
            rsi_period=indicator_cfg["rsi_period"],
            atr_period=indicator_cfg["atr_period"],
            bb_period=indicator_cfg["bb_period"],
            bb_std=indicator_cfg["bb_std"],
            session_ctx=session_ctx,
            recent_trades=self.risk_manager.recent_trades(),
        )

        if snapshot.gap and snapshot.gap.anomalous:
            self.logger.warning(
                f"Anomalous gap on {name}: size={snapshot.gap.size} "
                f"ratio={snapshot.gap.ratio}x ATR"
            )

        decision = await self.analyst.analyze(snapshot)
        if decision is None:
            self.logger.warning(f"No decision from Claude for {name}")
            return

        position_count = len([p for p in positions if p["size"] > 0])
        self.risk_manager.update_positions(position_count)
        self.risk_manager.update_balance(balance)
        self.risk_manager.sync_daily_pnl(float(daily_pnl))

        approved, reason = self.risk_manager.validate(decision)
        self.logger.info(f"Risk validation for {name}: approved={approved}, reason={reason}")

        gap_anomalous = bool(snapshot.gap and snapshot.gap.anomalous)
        blocked, block_reason = self._should_block_new_entry(session_ctx, gap_anomalous)
        if approved and decision.action.value != "HOLD" and blocked:
            self.logger.warning(f"Entry blocked for {name}: {block_reason}")
            return

        if approved and decision.action.value != "HOLD":
            trade = await self.executor.execute(decision, contract)
            if trade:
                self.logger.info(f"Trade executed for {name}: {trade}")
            else:
                self.logger.warning(f"Trade execution returned None for {name}")

    async def run_cycle(self):
        self.logger.info("-" * 40)
        self.logger.info("Starting analysis cycle")

        session = self.scheduler.get_session_info()
        self.logger.info(
            f"Session: {session['current_time']} | "
            f"Futures={'OPEN' if session['futures_open'] else 'CLOSED'} | "
            f"Forex={'OPEN' if session['forex_open'] else 'CLOSED'} | "
            f"mins_to_close={session['minutes_until_close']}"
        )

        weekend_buf = self.session_cfg.get("weekend_close_buffer_minutes", 30)
        flatten_weekend = self.session_cfg.get("flatten_into_weekend", True)
        if flatten_weekend and self.scheduler.is_near_weekend_close(weekend_buf):
            if not self._weekend_flattened:
                self.logger.warning("Near weekend close — flattening all positions")
                await self.executor.flatten_all_positions()
                await self.executor.cancel_all_orders()
                self._weekend_flattened = True
        else:
            self._weekend_flattened = False

        for name in self.contracts:
            try:
                await self.analyze_instrument(name)
            except Exception as e:
                self.logger.error(f"Error analyzing {name}: {e}", exc_info=True)

    async def run(self):
        await self.startup()

        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._signal_handler)

        self.logger.info(f"Trading loop started (interval={self.interval}s)")

        while not self._shutdown:
            try:
                await self.run_cycle()
            except Exception as e:
                self.logger.error(f"Cycle error: {e}", exc_info=True)

            if not self._shutdown:
                self.logger.info(f"Next cycle in {self.interval}s...")
                await asyncio.sleep(self.interval)

        await self.shutdown()

    def _signal_handler(self):
        self.logger.info("Shutdown signal received")
        self._shutdown = True

    async def shutdown(self):
        self.logger.info("=" * 50)
        self.logger.info("  Trading Agent Shutting Down")
        self.logger.info("=" * 50)

        open_trades = self.executor.get_open_orders()
        if open_trades:
            self.logger.info(f"Open trades at shutdown: {len(open_trades)}")

        await self.connection.disconnect()
        self.logger.info("Shutdown complete")


def main():
    agent = TradingAgent()
    asyncio.run(agent.run())


if __name__ == "__main__":
    main()
