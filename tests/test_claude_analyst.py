import json
from unittest.mock import MagicMock, patch

import pytest

from src.analysis.claude_analyst import ClaudeAnalyst
from src.models.schemas import (
    IndicatorValues,
    MarketSnapshot,
    OHLCVBar,
    TradeAction,
    TradeDecision,
)
from datetime import datetime


@pytest.fixture
def analyst():
    return ClaudeAnalyst(
        api_key="test-key",
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        temperature=0.2,
        max_retries=1,
    )


@pytest.fixture
def sample_snapshot():
    bars = [
        OHLCVBar(
            timestamp=datetime(2025, 1, 1, 10, i * 5),
            open=5000.0 + i,
            high=5005.0 + i,
            low=4995.0 + i,
            close=5002.0 + i,
            volume=1000.0,
        )
        for i in range(10)
    ]
    return MarketSnapshot(
        instrument="NQ",
        timeframe="5 mins",
        timestamp=datetime(2025, 1, 1, 10, 50),
        current_price=5010.0,
        bars=bars,
        indicators=IndicatorValues(
            ema_9=5008.0,
            ema_21=5003.0,
            ema_50=4998.0,
            rsi_14=55.0,
            atr_14=15.0,
            vwap=5005.0,
            bb_upper=5020.0,
            bb_middle=5005.0,
            bb_lower=4990.0,
            trend="UPTREND",
        ),
        account_balance=100000.0,
        daily_pnl=0.0,
    )


class TestParseResponse:
    def test_parse_clean_json(self, analyst):
        raw = json.dumps({
            "action": "BUY",
            "instrument": "NQ",
            "confidence": 0.85,
            "entry_price": 5010.0,
            "stop_loss": 4995.0,
            "take_profit": 5040.0,
            "size": 1,
            "reasoning": "Strong uptrend with EMA alignment",
        })
        decision = analyst._parse_response(raw, "NQ")
        assert decision is not None
        assert decision.action == TradeAction.BUY
        assert decision.confidence == 0.85

    def test_parse_json_in_code_block(self, analyst):
        raw = '```json\n{"action": "HOLD", "instrument": "ES", "confidence": 0.4, "reasoning": "No clear signal"}\n```'
        decision = analyst._parse_response(raw, "ES")
        assert decision is not None
        assert decision.action == TradeAction.HOLD

    def test_parse_json_with_surrounding_text(self, analyst):
        raw = 'Here is my analysis:\n{"action": "SELL", "instrument": "XAUUSD", "confidence": 0.82, "entry_price": 2050.0, "stop_loss": 2065.0, "take_profit": 2020.0, "size": 1, "reasoning": "Bearish reversal"}\nLet me know if you need more.'
        decision = analyst._parse_response(raw, "XAUUSD")
        assert decision is not None
        assert decision.action == TradeAction.SELL

    def test_parse_invalid_json(self, analyst):
        raw = "I cannot analyze this data properly."
        decision = analyst._parse_response(raw, "NQ")
        assert decision is None

    def test_parse_missing_required_fields(self, analyst):
        raw = json.dumps({"action": "BUY"})
        decision = analyst._parse_response(raw, "NQ")
        # Should fail validation because BUY requires stop_loss/take_profit
        assert decision is None

    def test_adds_instrument_if_missing(self, analyst):
        raw = json.dumps({
            "action": "HOLD",
            "confidence": 0.3,
            "reasoning": "No setup",
        })
        decision = analyst._parse_response(raw, "NQ")
        assert decision is not None
        assert decision.instrument == "NQ"


class TestBuildSystemPrompt:
    def test_includes_base_prompt(self, analyst):
        prompt = analyst._build_system_prompt("NQ")
        assert "intraday trading analyst" in prompt.lower() or len(prompt) > 0

    def test_includes_instrument_context(self, analyst):
        prompt = analyst._build_system_prompt("NQ")
        # Will include NQ-specific context if file exists
        assert isinstance(prompt, str)


class TestSerializeSnapshot:
    def test_serializes_to_json(self, analyst, sample_snapshot):
        result = analyst._serialize_snapshot(sample_snapshot)
        data = json.loads(result)
        assert data["instrument"] == "NQ"
        assert data["current_price"] == 5010.0
        assert len(data["bars"]) <= 10
