from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from ib_insync import (
    Contract,
    LimitOrder,
    MarketOrder,
    StopOrder,
    Trade,
)

from src.broker.connection import IBConnection
from src.models.schemas import TradeAction, TradeDecision, TradeOutcome
from src.risk.manager import RiskManager
from src.utils.logger import get_logger


@dataclass
class BracketContext:
    contract: Contract
    entry_action: TradeAction
    parent_trade: Trade
    stop_trade: Trade
    tp_trade: Trade
    reasoning: str = ""
    children_reconciled: bool = False
    result_recorded: bool = False


class TradeExecutor:
    """Places and manages orders through Interactive Brokers."""

    def __init__(self, connection: IBConnection, risk_manager: Optional[RiskManager] = None):
        self.conn = connection
        self.risk_manager = risk_manager
        self.logger = get_logger()
        self.brackets: dict[int, BracketContext] = {}

    async def execute(
        self,
        decision: TradeDecision,
        contract: Contract,
    ) -> Optional[Trade]:
        if not self.conn.connected:
            self.logger.error("Cannot execute: not connected to IB")
            return None

        if decision.action == TradeAction.HOLD:
            self.logger.info("Decision is HOLD -- no order placed")
            return None

        try:
            action = "BUY" if decision.action == TradeAction.BUY else "SELL"
            quantity = decision.size

            parent_order = MarketOrder(
                action=action,
                totalQuantity=quantity,
                transmit=False,
            )

            parent_trade = self.conn.ib.placeOrder(contract, parent_order)
            await asyncio.sleep(0.5)

            parent_id = parent_order.orderId
            sl_action = "SELL" if action == "BUY" else "BUY"

            stop_order = StopOrder(
                action=sl_action,
                totalQuantity=quantity,
                stopPrice=decision.stop_loss,
                parentId=parent_id,
                transmit=False,
            )
            stop_trade = self.conn.ib.placeOrder(contract, stop_order)

            tp_order = LimitOrder(
                action=sl_action,
                totalQuantity=quantity,
                lmtPrice=decision.take_profit,
                parentId=parent_id,
                transmit=True,
            )
            tp_trade = self.conn.ib.placeOrder(contract, tp_order)

            ctx = BracketContext(
                contract=contract,
                entry_action=decision.action,
                parent_trade=parent_trade,
                stop_trade=stop_trade,
                tp_trade=tp_trade,
                reasoning=decision.reasoning,
            )
            self.brackets[parent_id] = ctx
            self._attach_handlers(ctx)

            self.logger.info(
                f"Bracket order placed: {action} {quantity} {contract.symbol} | "
                f"SL={decision.stop_loss} | TP={decision.take_profit} | "
                f"Confidence={decision.confidence:.2f}"
            )
            self.logger.info(f"Reasoning: {decision.reasoning[:200]}")

            return parent_trade

        except Exception as e:
            self.logger.error(f"Order execution failed for {contract.symbol}: {e}")
            return None

    def _attach_handlers(self, ctx: BracketContext):
        ctx.parent_trade.statusEvent += lambda t, c=ctx: self._on_parent_status(t, c)
        ctx.stop_trade.filledEvent += lambda t, c=ctx: self._on_exit_filled(t, c)
        ctx.tp_trade.filledEvent += lambda t, c=ctx: self._on_exit_filled(t, c)

    def _on_parent_status(self, trade: Trade, ctx: BracketContext):
        """Reconcile bracket children when parent enters a terminal state.

        Why: IB does not automatically resize bracket children to match a partial
        parent fill. If we leave children at the original size, they would try to
        exit more contracts than we hold.
        """
        if ctx.children_reconciled:
            return

        status = trade.orderStatus.status
        if status not in ("Filled", "Cancelled", "ApiCancelled", "Inactive"):
            return

        filled = trade.orderStatus.filled
        requested = trade.order.totalQuantity

        if filled == 0:
            self.logger.warning(
                f"Parent order {trade.contract.symbol} ended with zero fills "
                f"(status={status}); cancelling brackets"
            )
            self._cancel_trade(ctx.stop_trade)
            self._cancel_trade(ctx.tp_trade)
            ctx.children_reconciled = True
            return

        if filled < requested:
            self.logger.warning(
                f"Partial fill on {trade.contract.symbol}: "
                f"{filled}/{requested} -- resizing brackets"
            )
            self._resize_child(ctx.stop_trade, filled)
            self._resize_child(ctx.tp_trade, filled)

        ctx.children_reconciled = True

    def _resize_child(self, child: Trade, new_qty: float):
        try:
            child.order.totalQuantity = new_qty
            self.conn.ib.placeOrder(child.contract, child.order)
        except Exception as e:
            self.logger.error(f"Failed to resize child order: {e}")

    def _cancel_trade(self, trade: Trade):
        try:
            self.conn.ib.cancelOrder(trade.order)
        except Exception as e:
            self.logger.error(f"Failed to cancel order: {e}")

    def _on_exit_filled(self, exit_trade: Trade, ctx: BracketContext):
        if ctx.result_recorded or self.risk_manager is None:
            return

        entry_price = ctx.parent_trade.orderStatus.avgFillPrice
        exit_price = exit_trade.orderStatus.avgFillPrice
        qty = exit_trade.orderStatus.filled

        if not entry_price or not exit_price or not qty:
            self.logger.warning("Exit fill missing price/qty data; skipping pnl record")
            return

        multiplier = float(getattr(ctx.contract, "multiplier", "") or 1)
        direction = 1 if ctx.entry_action == TradeAction.BUY else -1
        pnl = (exit_price - entry_price) * qty * multiplier * direction

        exit_reason = "STOP" if exit_trade is ctx.stop_trade else "TARGET"

        outcome = TradeOutcome(
            instrument=ctx.contract.symbol,
            action=ctx.entry_action.value,
            entry_time=datetime.utcnow(),  # IB Trade has fill log but utcnow is good enough here
            entry_price=float(entry_price),
            exit_time=datetime.utcnow(),
            exit_price=float(exit_price),
            pnl=pnl,
            exit_reason=exit_reason,
            reasoning=ctx.reasoning[:200],
        )

        self.logger.info(
            f"Exit filled on {ctx.contract.symbol}: "
            f"entry={entry_price} exit={exit_price} qty={qty} pnl=${pnl:.2f} "
            f"reason={exit_reason}"
        )
        self.risk_manager.record_trade_result(pnl, outcome=outcome)
        ctx.result_recorded = True

    def get_open_orders(self) -> list[Trade]:
        if not self.conn.connected:
            return []
        return self.conn.ib.openTrades()

    async def cancel_all_orders(self):
        if not self.conn.connected:
            return
        open_orders = self.conn.ib.openOrders()
        for order in open_orders:
            self.conn.ib.cancelOrder(order)
        self.logger.info(f"Cancelled {len(open_orders)} open orders")

    async def flatten_all_positions(self):
        """Close all open positions with market orders."""
        if not self.conn.connected:
            return

        positions = self.conn.ib.positions()
        for pos in positions:
            if pos.position == 0:
                continue
            action = "SELL" if pos.position > 0 else "BUY"
            quantity = abs(pos.position)
            order = MarketOrder(action=action, totalQuantity=quantity)
            self.conn.ib.placeOrder(pos.contract, order)
            self.logger.info(
                f"Flattening: {action} {quantity} {pos.contract.symbol}"
            )
