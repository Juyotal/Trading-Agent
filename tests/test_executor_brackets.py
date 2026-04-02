from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.broker.executor import BracketContext, TradeExecutor
from src.models.schemas import TradeAction


def make_trade(symbol="NQ", qty=2, status="Submitted", filled=0, avg_fill=0.0, multiplier="20"):
    contract = SimpleNamespace(symbol=symbol, multiplier=multiplier)
    order = SimpleNamespace(totalQuantity=qty, orderId=1)
    order_status = SimpleNamespace(status=status, filled=filled, avgFillPrice=avg_fill)
    return SimpleNamespace(contract=contract, order=order, orderStatus=order_status)


@pytest.fixture
def connection():
    ib = MagicMock()
    conn = SimpleNamespace(connected=True, ib=ib)
    return conn


@pytest.fixture
def risk_manager():
    rm = MagicMock()
    return rm


@pytest.fixture
def executor(connection, risk_manager):
    return TradeExecutor(connection, risk_manager)


def make_context(entry_action=TradeAction.BUY, qty=2, entry_fill=5000.0, parent_status="Filled", parent_filled=None):
    if parent_filled is None:
        parent_filled = qty
    parent = make_trade(qty=qty, status=parent_status, filled=parent_filled, avg_fill=entry_fill)
    stop = make_trade(qty=qty, status="PreSubmitted")
    tp = make_trade(qty=qty, status="PreSubmitted")
    return BracketContext(
        contract=parent.contract,
        entry_action=entry_action,
        parent_trade=parent,
        stop_trade=stop,
        tp_trade=tp,
    )


class TestParentStatusReconciliation:
    def test_full_fill_no_resize(self, executor, connection):
        ctx = make_context(qty=2, parent_filled=2, parent_status="Filled")
        executor._on_parent_status(ctx.parent_trade, ctx)
        assert ctx.children_reconciled is True
        # Children should not have been re-placed
        connection.ib.placeOrder.assert_not_called()
        connection.ib.cancelOrder.assert_not_called()

    def test_partial_fill_resizes_children(self, executor, connection):
        ctx = make_context(qty=3, parent_filled=2, parent_status="Filled")
        executor._on_parent_status(ctx.parent_trade, ctx)
        assert ctx.stop_trade.order.totalQuantity == 2
        assert ctx.tp_trade.order.totalQuantity == 2
        assert connection.ib.placeOrder.call_count == 2
        assert ctx.children_reconciled is True

    def test_zero_fill_cancels_children(self, executor, connection):
        ctx = make_context(qty=2, parent_filled=0, parent_status="Cancelled")
        executor._on_parent_status(ctx.parent_trade, ctx)
        assert connection.ib.cancelOrder.call_count == 2
        assert ctx.children_reconciled is True

    def test_non_terminal_status_ignored(self, executor, connection):
        ctx = make_context(parent_status="Submitted", parent_filled=0)
        executor._on_parent_status(ctx.parent_trade, ctx)
        assert ctx.children_reconciled is False
        connection.ib.placeOrder.assert_not_called()
        connection.ib.cancelOrder.assert_not_called()

    def test_reconcile_runs_only_once(self, executor, connection):
        ctx = make_context(qty=3, parent_filled=2, parent_status="Filled")
        executor._on_parent_status(ctx.parent_trade, ctx)
        executor._on_parent_status(ctx.parent_trade, ctx)
        # Still only one resize pass → 2 placeOrder calls total
        assert connection.ib.placeOrder.call_count == 2


class TestExitFillRecording:
    def test_winning_long_exit_records_positive_pnl(self, executor, risk_manager):
        ctx = make_context(entry_action=TradeAction.BUY, qty=2, entry_fill=5000.0)
        exit_trade = make_trade(qty=2, status="Filled", filled=2, avg_fill=5010.0, multiplier="20")
        executor._on_exit_filled(exit_trade, ctx)
        # (5010 - 5000) * 2 * 20 * 1 = +400
        call = risk_manager.record_trade_result.call_args
        assert call.args[0] == 400.0
        assert call.kwargs["outcome"].pnl == 400.0
        assert ctx.result_recorded is True

    def test_losing_short_exit_records_negative_pnl(self, executor, risk_manager):
        ctx = make_context(entry_action=TradeAction.SELL, qty=1, entry_fill=5000.0)
        exit_trade = make_trade(qty=1, status="Filled", filled=1, avg_fill=5010.0, multiplier="20")
        executor._on_exit_filled(exit_trade, ctx)
        # (5010 - 5000) * 1 * 20 * -1 = -200
        call = risk_manager.record_trade_result.call_args
        assert call.args[0] == -200.0
        assert call.kwargs["outcome"].action == "SELL"

    def test_no_multiplier_defaults_to_one(self, executor, risk_manager):
        ctx = make_context(entry_action=TradeAction.BUY, qty=1, entry_fill=2000.0)
        ctx.contract.multiplier = ""  # spot XAUUSD case
        exit_trade = make_trade(qty=1, status="Filled", filled=1, avg_fill=2010.0, multiplier="")
        executor._on_exit_filled(exit_trade, ctx)
        call = risk_manager.record_trade_result.call_args
        assert call.args[0] == 10.0

    def test_double_fire_records_only_once(self, executor, risk_manager):
        ctx = make_context(entry_action=TradeAction.BUY, qty=1, entry_fill=5000.0)
        exit_trade = make_trade(qty=1, status="Filled", filled=1, avg_fill=5005.0, multiplier="20")
        executor._on_exit_filled(exit_trade, ctx)
        executor._on_exit_filled(exit_trade, ctx)
        assert risk_manager.record_trade_result.call_count == 1

    def test_missing_fill_data_skips_record(self, executor, risk_manager):
        ctx = make_context(entry_action=TradeAction.BUY, qty=1, entry_fill=0.0)
        exit_trade = make_trade(qty=1, status="Filled", filled=0, avg_fill=0.0)
        executor._on_exit_filled(exit_trade, ctx)
        risk_manager.record_trade_result.assert_not_called()
        assert ctx.result_recorded is False

    def test_no_risk_manager_is_noop(self, connection):
        executor = TradeExecutor(connection, risk_manager=None)
        ctx = make_context(entry_action=TradeAction.BUY, qty=1, entry_fill=5000.0)
        exit_trade = make_trade(qty=1, status="Filled", filled=1, avg_fill=5010.0)
        executor._on_exit_filled(exit_trade, ctx)
        # Just shouldn't crash
