import pytest

from src.models.schemas import TradeAction, TradeDecision
from src.risk.manager import RiskManager, RiskState


@pytest.fixture
def risk_manager():
    return RiskManager(
        max_position_size=2,
        max_concurrent_positions=3,
        daily_loss_limit_pct=2.0,
        min_confidence=0.7,
        risk_per_trade_pct=1.0,
        max_consecutive_losses=5,
        account_balance=100_000.0,
    )


def make_decision(**kwargs) -> TradeDecision:
    defaults = {
        "action": TradeAction.BUY,
        "instrument": "NQ",
        "confidence": 0.85,
        "entry_price": 18000.0,
        "stop_loss": 17950.0,
        "take_profit": 18100.0,
        "size": 1,
        "reasoning": "Test trade",
    }
    defaults.update(kwargs)
    return TradeDecision(**defaults)


class TestConfidenceCheck:
    def test_high_confidence_passes(self, risk_manager):
        decision = make_decision(confidence=0.85)
        approved, _ = risk_manager.validate(decision)
        assert approved

    def test_low_confidence_rejected(self, risk_manager):
        decision = make_decision(confidence=0.5)
        approved, reason = risk_manager.validate(decision)
        assert not approved
        assert "Confidence" in reason

    def test_exact_threshold_passes(self, risk_manager):
        decision = make_decision(confidence=0.7)
        approved, _ = risk_manager.validate(decision)
        assert approved


class TestPositionLimits:
    def test_within_limits_passes(self, risk_manager):
        risk_manager.update_positions(2)
        decision = make_decision()
        approved, _ = risk_manager.validate(decision)
        assert approved

    def test_at_max_positions_rejected(self, risk_manager):
        risk_manager.update_positions(3)
        decision = make_decision()
        approved, reason = risk_manager.validate(decision)
        assert not approved
        assert "concurrent positions" in reason


class TestDailyLoss:
    def test_within_limit_passes(self, risk_manager):
        risk_manager.state.daily_pnl = -500.0
        decision = make_decision()
        approved, _ = risk_manager.validate(decision)
        assert approved

    def test_exceeding_limit_halts(self, risk_manager):
        risk_manager.state.daily_pnl = -2500.0
        decision = make_decision()
        approved, reason = risk_manager.validate(decision)
        assert not approved
        assert "Daily loss limit" in reason
        assert risk_manager.state.trading_halted


class TestConsecutiveLosses:
    def test_within_limit_passes(self, risk_manager):
        risk_manager.state.consecutive_losses = 3
        decision = make_decision()
        approved, _ = risk_manager.validate(decision)
        assert approved

    def test_max_losses_halts(self, risk_manager):
        risk_manager.state.consecutive_losses = 5
        decision = make_decision()
        approved, reason = risk_manager.validate(decision)
        assert not approved
        assert "consecutive losses" in reason


class TestStopLossDistance:
    def test_reasonable_stop_passes(self, risk_manager):
        decision = make_decision(entry_price=18000.0, stop_loss=17960.0)
        approved, _ = risk_manager.validate(decision)
        assert approved

    def test_too_wide_stop_rejected(self, risk_manager):
        decision = make_decision(entry_price=18000.0, stop_loss=17000.0)
        approved, reason = risk_manager.validate(decision)
        assert not approved
        assert "Stop loss too wide" in reason


class TestPositionSize:
    def test_valid_size_passes(self, risk_manager):
        decision = make_decision(size=1)
        approved, _ = risk_manager.validate(decision)
        assert approved

    def test_oversized_rejected(self, risk_manager):
        decision = make_decision(size=5)
        approved, reason = risk_manager.validate(decision)
        assert not approved
        assert "Position size" in reason


class TestHoldAction:
    def test_hold_always_passes(self, risk_manager):
        decision = TradeDecision(
            action=TradeAction.HOLD,
            instrument="NQ",
            confidence=0.3,
            reasoning="No clear setup",
        )
        approved, reason = risk_manager.validate(decision)
        assert approved
        assert "HOLD" in reason


class TestTradeRecording:
    def test_winning_trade_resets_losses(self, risk_manager):
        risk_manager.state.consecutive_losses = 3
        risk_manager.record_trade_result(500.0)
        assert risk_manager.state.consecutive_losses == 0
        assert risk_manager.state.daily_pnl == 500.0

    def test_losing_trade_increments(self, risk_manager):
        risk_manager.record_trade_result(-200.0)
        risk_manager.record_trade_result(-150.0)
        assert risk_manager.state.consecutive_losses == 2
        assert risk_manager.state.daily_pnl == -350.0
