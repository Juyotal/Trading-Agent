from datetime import datetime, timedelta

import pandas as pd
import pytest

from src.backtest.fill_sim import simulate_bracket
from src.models.schemas import TradeAction, TradeDecision


def make_bars(prices: list[tuple[float, float, float, float]], start="2026-01-01 09:30"):
    """prices = list of (open, high, low, close) tuples. One bar per tuple."""
    ts = pd.date_range(start, periods=len(prices), freq="5min")
    return pd.DataFrame({
        "timestamp": ts,
        "open": [p[0] for p in prices],
        "high": [p[1] for p in prices],
        "low": [p[2] for p in prices],
        "close": [p[3] for p in prices],
        "volume": [1000] * len(prices),
    })


def buy_decision(entry=5000, sl=4980, tp=5040):
    return TradeDecision(
        action=TradeAction.BUY,
        instrument="NQ",
        confidence=0.8,
        entry_price=entry,
        stop_loss=sl,
        take_profit=tp,
        size=1,
        reasoning="test",
    )


def sell_decision(entry=5000, sl=5020, tp=4960):
    return TradeDecision(
        action=TradeAction.SELL,
        instrument="NQ",
        confidence=0.8,
        entry_price=entry,
        stop_loss=sl,
        take_profit=tp,
        size=1,
        reasoning="test",
    )


class TestLongTrades:
    def test_target_hit(self):
        bars = make_bars([
            (5000, 5001, 4999, 5000),  # signal bar (not used for entry)
            (5000, 5010, 4995, 5005),  # entry at 5000
            (5005, 5045, 5000, 5040),  # TP 5040 hit
        ])
        decision = buy_decision(sl=4980, tp=5040)
        trade = simulate_bracket(decision, bars.iloc[0:], multiplier=20)
        assert trade.exit_reason == "TARGET"
        assert trade.exit_price == 5040
        # (5040 - 5000) * 1 * 20 = 800
        assert trade.pnl == 800.0

    def test_stop_hit(self):
        bars = make_bars([
            (5000, 5001, 4999, 5000),
            (5000, 5005, 4990, 4995),
            (4995, 5000, 4975, 4980),  # SL 4980 hit
        ])
        decision = buy_decision(sl=4980, tp=5040)
        trade = simulate_bracket(decision, bars.iloc[0:], multiplier=20)
        assert trade.exit_reason == "STOP"
        assert trade.exit_price == 4980
        assert trade.pnl == -400.0

    def test_both_hit_same_bar_assumes_stop(self):
        bars = make_bars([
            (5000, 5001, 4999, 5000),
            (5000, 5010, 4995, 5005),
            (5005, 5045, 4975, 5000),  # both TP and SL traded
        ])
        decision = buy_decision(sl=4980, tp=5040)
        trade = simulate_bracket(decision, bars.iloc[0:], multiplier=20)
        assert trade.exit_reason == "STOP"
        assert trade.exit_price == 4980

    def test_gap_down_below_stop_fills_at_open(self):
        bars = make_bars([
            (5000, 5001, 4999, 5000),
            (5000, 5005, 4990, 4995),
            (4950, 4960, 4940, 4945),  # gap opens below SL 4980
        ])
        decision = buy_decision(sl=4980, tp=5040)
        trade = simulate_bracket(decision, bars.iloc[0:], multiplier=20)
        assert trade.exit_reason == "STOP"
        assert trade.exit_price == 4950  # filled at open, worse than SL

    def test_no_exit_closes_at_end(self):
        bars = make_bars([
            (5000, 5001, 4999, 5000),
            (5000, 5010, 4995, 5005),
            (5005, 5015, 4990, 5010),
            (5010, 5020, 5000, 5015),  # within range, never hits SL or TP
        ])
        decision = buy_decision(sl=4980, tp=5050)
        trade = simulate_bracket(decision, bars.iloc[0:], multiplier=20)
        assert trade.exit_reason == "END_OF_DATA"
        assert trade.exit_price == 5015


class TestShortTrades:
    def test_target_hit(self):
        bars = make_bars([
            (5000, 5001, 4999, 5000),
            (5000, 5005, 4990, 4995),
            (4995, 5000, 4955, 4960),  # TP 4960 hit
        ])
        decision = sell_decision(sl=5020, tp=4960)
        trade = simulate_bracket(decision, bars.iloc[0:], multiplier=20)
        assert trade.exit_reason == "TARGET"
        assert trade.exit_price == 4960
        # short: (4960 - 5000) * 1 * 20 * -1 = 800
        assert trade.pnl == 800.0

    def test_stop_hit(self):
        bars = make_bars([
            (5000, 5001, 4999, 5000),
            (5000, 5010, 4995, 5005),
            (5005, 5025, 5000, 5020),  # SL 5020 hit
        ])
        decision = sell_decision(sl=5020, tp=4960)
        trade = simulate_bracket(decision, bars.iloc[0:], multiplier=20)
        assert trade.exit_reason == "STOP"
        assert trade.exit_price == 5020
        assert trade.pnl == -400.0

    def test_gap_up_above_stop_fills_at_open(self):
        bars = make_bars([
            (5000, 5001, 4999, 5000),
            (5000, 5010, 4995, 5005),
            (5060, 5070, 5050, 5065),  # gap above SL 5020
        ])
        decision = sell_decision(sl=5020, tp=4960)
        trade = simulate_bracket(decision, bars.iloc[0:], multiplier=20)
        assert trade.exit_reason == "STOP"
        assert trade.exit_price == 5060


class TestEdgeCases:
    def test_hold_returns_none(self):
        bars = make_bars([(5000, 5001, 4999, 5000), (5000, 5010, 4995, 5005)])
        decision = TradeDecision(
            action=TradeAction.HOLD,
            instrument="NQ",
            confidence=0.3,
            reasoning="test",
        )
        assert simulate_bracket(decision, bars, multiplier=20) is None

    def test_insufficient_future_bars_returns_none(self):
        bars = make_bars([(5000, 5001, 4999, 5000)])
        decision = buy_decision()
        assert simulate_bracket(decision, bars, multiplier=20) is None
