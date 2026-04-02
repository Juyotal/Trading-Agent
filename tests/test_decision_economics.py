import pytest

from src.analysis.claude_analyst import validate_decision_economics
from src.models.schemas import TradeAction, TradeDecision


def make(action, entry=5000, sl=None, tp=None):
    return TradeDecision(
        action=action,
        instrument="NQ",
        confidence=0.8,
        entry_price=entry,
        stop_loss=sl,
        take_profit=tp,
        size=1,
        reasoning="test",
    )


class TestHold:
    def test_hold_passes_without_levels(self):
        d = TradeDecision(action=TradeAction.HOLD, instrument="NQ", confidence=0.3, reasoning="hold")
        ok, _ = validate_decision_economics(d, 5000)
        assert ok


class TestLong:
    def test_valid_long(self):
        d = make(TradeAction.BUY, entry=5000, sl=4980, tp=5040)
        ok, reason = validate_decision_economics(d, 5000)
        assert ok, reason

    def test_stop_above_entry_rejected(self):
        d = make(TradeAction.BUY, entry=5000, sl=5020, tp=5040)
        ok, reason = validate_decision_economics(d, 5000)
        assert not ok
        assert "stop" in reason.lower()

    def test_target_below_entry_rejected(self):
        d = make(TradeAction.BUY, entry=5000, sl=4980, tp=4990)
        ok, reason = validate_decision_economics(d, 5000)
        assert not ok
        assert "target" in reason.lower()

    def test_target_equal_to_entry_rejected(self):
        d = make(TradeAction.BUY, entry=5000, sl=4980, tp=5000)
        ok, _ = validate_decision_economics(d, 5000)
        assert not ok

    def test_low_reward_risk_rejected(self):
        # risk=20, reward=5 → RR=0.25, below default min 0.5
        d = make(TradeAction.BUY, entry=5000, sl=4980, tp=5005)
        ok, reason = validate_decision_economics(d, 5000)
        assert not ok
        assert "reward:risk" in reason


class TestShort:
    def test_valid_short(self):
        d = make(TradeAction.SELL, entry=5000, sl=5020, tp=4960)
        ok, reason = validate_decision_economics(d, 5000)
        assert ok, reason

    def test_stop_below_entry_rejected(self):
        d = make(TradeAction.SELL, entry=5000, sl=4980, tp=4960)
        ok, _ = validate_decision_economics(d, 5000)
        assert not ok

    def test_target_above_entry_rejected(self):
        d = make(TradeAction.SELL, entry=5000, sl=5020, tp=5010)
        ok, _ = validate_decision_economics(d, 5000)
        assert not ok


class TestEntryDrift:
    def test_entry_within_drift_window(self):
        # Entry 5010 with current 5000 → 0.2% drift, within 2% default
        d = make(TradeAction.BUY, entry=5010, sl=4990, tp=5050)
        ok, _ = validate_decision_economics(d, 5000)
        assert ok

    def test_entry_too_far_from_current_rejected(self):
        # Entry 5200 with current 5000 → 4% drift, exceeds 2% default
        d = make(TradeAction.BUY, entry=5200, sl=5180, tp=5240)
        ok, reason = validate_decision_economics(d, 5000)
        assert not ok
        assert "drift" in reason.lower()


class TestMissingFields:
    def test_none_entry_rejected(self):
        d = make(TradeAction.BUY, entry=None, sl=4980, tp=5040)
        # Pydantic won't allow None entry when the validator runs for BUY — but schema
        # allows entry_price=None. Construct manually.
        d = TradeDecision.model_construct(
            action=TradeAction.BUY, instrument="NQ", confidence=0.8,
            entry_price=None, stop_loss=4980, take_profit=5040, size=1, reasoning="x",
        )
        ok, reason = validate_decision_economics(d, 5000)
        assert not ok
        assert "missing" in reason.lower()
