from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

from src.models.schemas import TradeAction, TradeDecision, TradeOutcome
from src.utils.logger import get_logger


@dataclass
class RiskState:
    daily_pnl: float = 0.0
    daily_trades: int = 0
    consecutive_losses: int = 0
    open_positions: int = 0
    trading_halted: bool = False
    halt_reason: str = ""
    today: date = field(default_factory=date.today)
    recent_trades: deque = field(default_factory=lambda: deque(maxlen=20))

    def reset_if_new_day(self):
        if date.today() != self.today:
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.consecutive_losses = 0
            self.trading_halted = False
            self.halt_reason = ""
            self.today = date.today()


class RiskManager:
    """Validates trade decisions against risk rules before execution."""

    def __init__(
        self,
        max_position_size: int = 1,
        max_concurrent_positions: int = 3,
        daily_loss_limit_pct: float = 2.0,
        min_confidence: float = 0.7,
        risk_per_trade_pct: float = 1.0,
        max_consecutive_losses: int = 5,
        account_balance: float = 100_000.0,
        memory_path: Optional[str] = None,
        memory_size: int = 20,
    ):
        self.max_position_size = max_position_size
        self.max_concurrent_positions = max_concurrent_positions
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.min_confidence = min_confidence
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_consecutive_losses = max_consecutive_losses
        self.account_balance = account_balance
        self.state = RiskState(recent_trades=deque(maxlen=memory_size))
        self.memory_path = Path(memory_path) if memory_path else None
        self.logger = get_logger()
        self._load_memory()

    def validate(self, decision: TradeDecision) -> tuple[bool, str]:
        self.state.reset_if_new_day()

        if self.state.trading_halted:
            return False, f"Trading halted: {self.state.halt_reason}"

        if decision.action == TradeAction.HOLD:
            return True, "HOLD -- no action needed"

        checks = [
            self._check_confidence(decision),
            self._check_position_limits(decision),
            self._check_daily_loss(),
            self._check_consecutive_losses(),
            self._check_stop_loss_distance(decision),
            self._check_position_size(decision),
        ]

        for passed, reason in checks:
            if not passed:
                self.logger.warning(f"Risk check FAILED: {reason}")
                return False, reason

        self.logger.info(f"Risk checks PASSED for {decision.instrument} {decision.action.value}")
        return True, "All risk checks passed"

    def _check_confidence(self, decision: TradeDecision) -> tuple[bool, str]:
        if decision.confidence < self.min_confidence:
            return False, (
                f"Confidence {decision.confidence:.2f} below threshold "
                f"{self.min_confidence:.2f}"
            )
        return True, ""

    def _check_position_limits(self, decision: TradeDecision) -> tuple[bool, str]:
        if self.state.open_positions >= self.max_concurrent_positions:
            return False, (
                f"Max concurrent positions reached "
                f"({self.state.open_positions}/{self.max_concurrent_positions})"
            )
        return True, ""

    def _check_daily_loss(self) -> tuple[bool, str]:
        daily_loss_limit = self.account_balance * (self.daily_loss_limit_pct / 100)
        if self.state.daily_pnl < -daily_loss_limit:
            self.state.trading_halted = True
            self.state.halt_reason = (
                f"Daily loss limit reached: ${self.state.daily_pnl:.2f} "
                f"(limit: -${daily_loss_limit:.2f})"
            )
            return False, self.state.halt_reason
        return True, ""

    def _check_consecutive_losses(self) -> tuple[bool, str]:
        if self.state.consecutive_losses >= self.max_consecutive_losses:
            self.state.trading_halted = True
            self.state.halt_reason = (
                f"Max consecutive losses reached: {self.state.consecutive_losses}"
            )
            return False, self.state.halt_reason
        return True, ""

    def _check_stop_loss_distance(self, decision: TradeDecision) -> tuple[bool, str]:
        if decision.entry_price and decision.stop_loss:
            distance_pct = abs(decision.entry_price - decision.stop_loss) / decision.entry_price * 100
            max_risk = self.risk_per_trade_pct * 2
            if distance_pct > max_risk:
                return False, (
                    f"Stop loss too wide: {distance_pct:.2f}% "
                    f"(max {max_risk:.2f}%)"
                )
        return True, ""

    def _check_position_size(self, decision: TradeDecision) -> tuple[bool, str]:
        if decision.size > self.max_position_size:
            return False, (
                f"Position size {decision.size} exceeds max {self.max_position_size}"
            )
        return True, ""

    def record_trade_result(self, pnl: float, outcome: Optional[TradeOutcome] = None):
        self.state.daily_pnl += pnl
        self.state.daily_trades += 1
        if pnl < 0:
            self.state.consecutive_losses += 1
        else:
            self.state.consecutive_losses = 0
        if outcome is not None:
            self.state.recent_trades.append(outcome)
            self._persist_memory()
        self.logger.info(
            f"Trade result recorded: pnl=${pnl:.2f} | "
            f"daily_pnl=${self.state.daily_pnl:.2f} | "
            f"consecutive_losses={self.state.consecutive_losses}"
        )

    def recent_trades(self) -> list[TradeOutcome]:
        return list(self.state.recent_trades)

    def _load_memory(self):
        if self.memory_path is None or not self.memory_path.exists():
            return
        try:
            for line in self.memory_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                self.state.recent_trades.append(TradeOutcome(**json.loads(line)))
            self.logger.info(
                f"Loaded {len(self.state.recent_trades)} trade outcomes from {self.memory_path}"
            )
        except Exception as e:
            self.logger.warning(f"Failed to load trade memory from {self.memory_path}: {e}")

    def _persist_memory(self):
        if self.memory_path is None:
            return
        try:
            lines = [t.model_dump_json() for t in self.state.recent_trades]
            self.memory_path.write_text("\n".join(lines) + "\n")
        except Exception as e:
            self.logger.warning(f"Failed to persist trade memory to {self.memory_path}: {e}")

    def sync_daily_pnl(self, realized_pnl: float):
        """Sync daily PnL from the broker's realized PnL figure.

        Why: record_trade_result is fill-event driven and can miss events across
        restarts or reconnects. IB's RealizedPnL is authoritative for the day.
        """
        self.state.reset_if_new_day()
        self.state.daily_pnl = realized_pnl

    def update_positions(self, count: int):
        self.state.open_positions = count

    def update_balance(self, balance: float):
        self.account_balance = balance
