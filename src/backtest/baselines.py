from __future__ import annotations

import random
from typing import Optional

import pandas as pd

from src.models.schemas import MarketSnapshot, TradeAction, TradeDecision


class RandomAnalyst:
    """Random-entry baseline. Decides BUY/SELL/HOLD uniformly at random.

    Uses ATR to size stop and target symmetrically so it matches the shape of
    a real strategy — just without any signal. Beats this and you have a signal.
    """

    def __init__(self, trade_prob: float = 0.2, atr_mult_stop: float = 1.5, atr_mult_target: float = 2.0, seed: int = 42):
        self.trade_prob = trade_prob
        self.atr_mult_stop = atr_mult_stop
        self.atr_mult_target = atr_mult_target
        self.rng = random.Random(seed)

    async def analyze(self, snapshot: MarketSnapshot) -> Optional[TradeDecision]:
        atr = snapshot.indicators.atr_14
        price = snapshot.current_price
        if atr is None or atr <= 0:
            return TradeDecision(action=TradeAction.HOLD, instrument=snapshot.instrument,
                                 confidence=0.0, reasoning="no ATR")

        if self.rng.random() > self.trade_prob:
            return TradeDecision(action=TradeAction.HOLD, instrument=snapshot.instrument,
                                 confidence=0.0, reasoning="random HOLD")

        direction = self.rng.choice([TradeAction.BUY, TradeAction.SELL])
        if direction == TradeAction.BUY:
            sl = price - atr * self.atr_mult_stop
            tp = price + atr * self.atr_mult_target
        else:
            sl = price + atr * self.atr_mult_stop
            tp = price - atr * self.atr_mult_target

        return TradeDecision(
            action=direction,
            instrument=snapshot.instrument,
            confidence=0.75,  # above the 0.7 gate so trades actually execute
            entry_price=price,
            stop_loss=round(sl, 4),
            take_profit=round(tp, 4),
            size=1,
            reasoning="random baseline",
        )


class SmaCrossoverAnalyst:
    """Classic SMA(9/21) crossover baseline. Long when 9 > 21, short when 9 < 21.

    Signals only on the crossover bar to avoid constant flipping. Uses the same
    ATR-scaled stops/targets as the random baseline for apples-to-apples.
    """

    def __init__(self, fast: int = 9, slow: int = 21, atr_mult_stop: float = 1.5, atr_mult_target: float = 2.0):
        self.fast = fast
        self.slow = slow
        self.atr_mult_stop = atr_mult_stop
        self.atr_mult_target = atr_mult_target

    async def analyze(self, snapshot: MarketSnapshot) -> Optional[TradeDecision]:
        atr = snapshot.indicators.atr_14
        price = snapshot.current_price
        if atr is None or atr <= 0 or len(snapshot.bars) < self.slow + 1:
            return TradeDecision(action=TradeAction.HOLD, instrument=snapshot.instrument,
                                 confidence=0.0, reasoning="insufficient data")

        closes = pd.Series([b.close for b in snapshot.bars])
        fast_now = closes.rolling(self.fast).mean().iloc[-1]
        fast_prev = closes.rolling(self.fast).mean().iloc[-2]
        slow_now = closes.rolling(self.slow).mean().iloc[-1]
        slow_prev = closes.rolling(self.slow).mean().iloc[-2]

        if pd.isna(fast_now) or pd.isna(slow_now) or pd.isna(fast_prev) or pd.isna(slow_prev):
            return TradeDecision(action=TradeAction.HOLD, instrument=snapshot.instrument,
                                 confidence=0.0, reasoning="SMA warmup")

        crossed_up = fast_prev <= slow_prev and fast_now > slow_now
        crossed_down = fast_prev >= slow_prev and fast_now < slow_now

        if crossed_up:
            return TradeDecision(
                action=TradeAction.BUY,
                instrument=snapshot.instrument,
                confidence=0.75,
                entry_price=price,
                stop_loss=round(price - atr * self.atr_mult_stop, 4),
                take_profit=round(price + atr * self.atr_mult_target, 4),
                size=1,
                reasoning=f"SMA{self.fast} crossed above SMA{self.slow}",
            )
        if crossed_down:
            return TradeDecision(
                action=TradeAction.SELL,
                instrument=snapshot.instrument,
                confidence=0.75,
                entry_price=price,
                stop_loss=round(price + atr * self.atr_mult_stop, 4),
                take_profit=round(price - atr * self.atr_mult_target, 4),
                size=1,
                reasoning=f"SMA{self.fast} crossed below SMA{self.slow}",
            )

        return TradeDecision(action=TradeAction.HOLD, instrument=snapshot.instrument,
                             confidence=0.0, reasoning="no crossover")


def buy_and_hold_pnl(df: pd.DataFrame, multiplier: float = 1.0) -> float:
    """Reference: hold 1 unit from first bar's open to last bar's close."""
    if len(df) < 2:
        return 0.0
    entry = float(df.iloc[0]["open"])
    exit_price = float(df.iloc[-1]["close"])
    return (exit_price - entry) * multiplier
