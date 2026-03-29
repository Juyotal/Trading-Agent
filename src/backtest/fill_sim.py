from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd

from src.models.schemas import TradeAction, TradeDecision


@dataclass
class SimulatedTrade:
    instrument: str
    action: TradeAction
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    size: int
    stop_loss: float
    take_profit: float
    exit_reason: str  # "STOP", "TARGET", "END_OF_DATA"
    pnl: float
    confidence: float
    reasoning: str


def simulate_bracket(
    decision: TradeDecision,
    future_bars: pd.DataFrame,
    multiplier: float = 1.0,
    decision_time: Optional[datetime] = None,
) -> Optional[SimulatedTrade]:
    """Replay a bracket order against future bars.

    Entry fills at the next bar's open. Subsequent bars are scanned for SL/TP
    hits using high/low. If both levels trade in the same bar, assume the stop
    fills first (pessimistic).
    """
    if len(future_bars) < 2 or decision.action == TradeAction.HOLD:
        return None

    entry_bar = future_bars.iloc[0]
    entry_price = float(entry_bar["open"])
    entry_time = entry_bar["timestamp"]

    sl = decision.stop_loss
    tp = decision.take_profit
    is_long = decision.action == TradeAction.BUY
    direction = 1 if is_long else -1

    exit_price = None
    exit_time = None
    exit_reason = "END_OF_DATA"

    for _, bar in future_bars.iloc[1:].iterrows():
        high = float(bar["high"])
        low = float(bar["low"])
        bar_open = float(bar["open"])

        if is_long:
            hit_stop = low <= sl
            hit_target = high >= tp
            if hit_stop and hit_target:
                # Pessimistic: stop hits first. Account for gap-down open below SL.
                exit_price = min(bar_open, sl) if bar_open <= sl else sl
                exit_reason = "STOP"
            elif hit_stop:
                exit_price = min(bar_open, sl) if bar_open <= sl else sl
                exit_reason = "STOP"
            elif hit_target:
                exit_price = max(bar_open, tp) if bar_open >= tp else tp
                exit_reason = "TARGET"
        else:
            hit_stop = high >= sl
            hit_target = low <= tp
            if hit_stop and hit_target:
                exit_price = max(bar_open, sl) if bar_open >= sl else sl
                exit_reason = "STOP"
            elif hit_stop:
                exit_price = max(bar_open, sl) if bar_open >= sl else sl
                exit_reason = "STOP"
            elif hit_target:
                exit_price = min(bar_open, tp) if bar_open <= tp else tp
                exit_reason = "TARGET"

        if exit_price is not None:
            exit_time = bar["timestamp"]
            break

    if exit_price is None:
        last = future_bars.iloc[-1]
        exit_price = float(last["close"])
        exit_time = last["timestamp"]
        exit_reason = "END_OF_DATA"

    pnl = (exit_price - entry_price) * decision.size * multiplier * direction

    return SimulatedTrade(
        instrument=decision.instrument,
        action=decision.action,
        entry_time=entry_time,
        entry_price=entry_price,
        exit_time=exit_time,
        exit_price=exit_price,
        size=decision.size,
        stop_loss=sl,
        take_profit=tp,
        exit_reason=exit_reason,
        pnl=pnl,
        confidence=decision.confidence,
        reasoning=decision.reasoning,
    )
