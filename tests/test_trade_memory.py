from datetime import datetime

import pytest

from src.models.schemas import TradeOutcome
from src.risk.manager import RiskManager


def make_outcome(pnl=100.0, instrument="MNQ"):
    return TradeOutcome(
        instrument=instrument,
        action="BUY",
        entry_time=datetime(2026, 4, 17, 10, 0),
        entry_price=5000.0,
        exit_time=datetime(2026, 4, 17, 10, 30),
        exit_price=5005.0,
        pnl=pnl,
        exit_reason="TARGET",
        reasoning="test trade",
    )


class TestBuffer:
    def test_empty_by_default(self):
        rm = RiskManager(account_balance=100_000.0)
        assert rm.recent_trades() == []

    def test_appends_outcome(self):
        rm = RiskManager(account_balance=100_000.0)
        rm.record_trade_result(100.0, outcome=make_outcome(pnl=100.0))
        assert len(rm.recent_trades()) == 1
        assert rm.recent_trades()[0].pnl == 100.0

    def test_pnl_only_does_not_append(self):
        rm = RiskManager(account_balance=100_000.0)
        rm.record_trade_result(100.0)
        assert rm.recent_trades() == []

    def test_rolling_window(self):
        rm = RiskManager(account_balance=100_000.0, memory_size=3)
        for i in range(5):
            rm.record_trade_result(i * 10.0, outcome=make_outcome(pnl=i * 10.0))
        trades = rm.recent_trades()
        assert len(trades) == 3
        # oldest two dropped → pnl values 20, 30, 40
        assert [t.pnl for t in trades] == [20.0, 30.0, 40.0]


class TestPersistence:
    def test_writes_and_reads_memory_file(self, tmp_path):
        mem = tmp_path / "mem.jsonl"

        rm1 = RiskManager(account_balance=100_000.0, memory_path=str(mem), memory_size=10)
        rm1.record_trade_result(50.0, outcome=make_outcome(pnl=50.0))
        rm1.record_trade_result(-25.0, outcome=make_outcome(pnl=-25.0, instrument="MES"))

        assert mem.exists()

        # Fresh instance should load what was persisted
        rm2 = RiskManager(account_balance=100_000.0, memory_path=str(mem), memory_size=10)
        loaded = rm2.recent_trades()
        assert len(loaded) == 2
        assert loaded[0].pnl == 50.0
        assert loaded[1].pnl == -25.0
        assert loaded[1].instrument == "MES"

    def test_no_persistence_when_path_is_none(self, tmp_path):
        rm = RiskManager(account_balance=100_000.0, memory_path=None)
        rm.record_trade_result(50.0, outcome=make_outcome())
        # Just shouldn't crash and shouldn't touch disk
        assert len(list(tmp_path.iterdir())) == 0

    def test_corrupt_memory_file_does_not_crash(self, tmp_path):
        mem = tmp_path / "mem.jsonl"
        mem.write_text("not valid json\n{also not valid\n")
        rm = RiskManager(account_balance=100_000.0, memory_path=str(mem))
        # Should log a warning and start empty, not raise
        assert rm.recent_trades() == []


class TestSerializationIntoSnapshot:
    def test_outcomes_serialize_cleanly(self):
        outcome = make_outcome()
        dumped = outcome.model_dump_json()
        restored = TradeOutcome.model_validate_json(dumped)
        assert restored.pnl == outcome.pnl
        assert restored.exit_reason == "TARGET"
