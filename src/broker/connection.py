from __future__ import annotations

import asyncio
import time
from typing import Optional

from ib_insync import IB, Contract, util

from src.utils.logger import get_logger


class IBConnection:
    """Manages connection lifecycle to Interactive Brokers Gateway/TWS."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 1,
        timeout: int = 30,
        auto_reconnect: bool = True,
    ):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.timeout = timeout
        self.auto_reconnect = auto_reconnect
        self.ib = IB()
        self._connected = False
        self.logger = get_logger()

    async def connect(self) -> bool:
        try:
            self.logger.info(
                f"Connecting to IB Gateway at {self.host}:{self.port} "
                f"(client_id={self.client_id})"
            )
            await self.ib.connectAsync(
                self.host, self.port, clientId=self.client_id, timeout=self.timeout
            )
            self._connected = True
            self.logger.info("Connected to IB Gateway successfully")

            if self.auto_reconnect:
                self.ib.disconnectedEvent += self._on_disconnect

            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to IB Gateway: {e}")
            self._connected = False
            return False

    def _on_disconnect(self):
        self.logger.warning("Disconnected from IB Gateway")
        self._connected = False
        if self.auto_reconnect:
            self.logger.info("Auto-reconnect enabled, attempting reconnect in 5s...")
            asyncio.get_event_loop().call_later(5, lambda: asyncio.ensure_future(self._reconnect()))

    async def _reconnect(self, max_attempts: int = 5):
        for attempt in range(1, max_attempts + 1):
            self.logger.info(f"Reconnect attempt {attempt}/{max_attempts}")
            if await self.connect():
                return
            await asyncio.sleep(min(5 * attempt, 30))
        self.logger.error("All reconnect attempts failed")

    async def disconnect(self):
        if self._connected:
            self.ib.disconnectedEvent.clear()
            self.ib.disconnect()
            self._connected = False
            self.logger.info("Disconnected from IB Gateway")

    @property
    def connected(self) -> bool:
        return self._connected and self.ib.isConnected()

    def get_account_summary(self) -> dict:
        if not self.connected:
            return {}
        values = self.ib.accountSummary()
        summary = {}
        for v in values:
            if v.tag in ("NetLiquidation", "TotalCashValue", "UnrealizedPnL", "RealizedPnL"):
                summary[v.tag] = float(v.value)
        return summary

    def make_contract(self, symbol: str, exchange: str, sec_type: str, currency: str = "USD") -> Contract:
        if sec_type == "FUT":
            contract = Contract(
                symbol=symbol,
                exchange=exchange,
                secType="FUT",
                currency=currency,
                lastTradeDateOrContractMonth="",  # will be resolved by qualifyContracts
            )
        elif sec_type == "CASH":
            base = symbol[:3] if len(symbol) == 6 else "XAU"
            quote = symbol[3:] if len(symbol) == 6 else "USD"
            contract = Contract(
                symbol=base,
                exchange=exchange,
                secType="CASH",
                currency=quote,
            )
        else:
            contract = Contract(
                symbol=symbol,
                exchange=exchange,
                secType=sec_type,
                currency=currency,
            )
        return contract

    async def qualify_contract(self, contract: Contract) -> Optional[Contract]:
        try:
            qualified = await self.ib.qualifyContractsAsync(contract)
            if qualified:
                return qualified[0]
            self.logger.warning(f"Could not qualify contract: {contract}")
            return None
        except Exception as e:
            self.logger.error(f"Error qualifying contract {contract}: {e}")
            return None
