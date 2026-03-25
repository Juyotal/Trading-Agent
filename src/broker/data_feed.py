from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Optional

import pandas as pd
from ib_insync import Contract, util

from src.broker.connection import IBConnection
from src.utils.logger import get_logger


class DataFeed:
    """Fetches historical and real-time market data from IB."""

    def __init__(self, connection: IBConnection):
        self.conn = connection
        self.logger = get_logger()

    async def get_historical_bars(
        self,
        contract: Contract,
        bar_size: str = "5 mins",
        duration: str = "1 D",
        what_to_show: str = "TRADES",
    ) -> Optional[pd.DataFrame]:
        if not self.conn.connected:
            self.logger.error("Cannot fetch data: not connected to IB")
            return None

        try:
            bars = await self.conn.ib.reqHistoricalDataAsync(
                contract,
                endDateTime="",
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=False,
                formatDate=1,
            )

            if not bars:
                self.logger.warning(f"No bars returned for {contract.symbol}")
                return None

            df = util.df(bars)
            df = df.rename(columns={
                "date": "timestamp",
            })
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df[["timestamp", "open", "high", "low", "close", "volume"]]
            df = df.sort_values("timestamp").reset_index(drop=True)

            self.logger.info(
                f"Fetched {len(df)} bars for {contract.symbol} "
                f"({bar_size}, {duration})"
            )
            return df

        except Exception as e:
            self.logger.error(f"Error fetching historical data for {contract.symbol}: {e}")
            return None

    async def get_current_price(self, contract: Contract) -> Optional[float]:
        if not self.conn.connected:
            return None

        try:
            ticker = self.conn.ib.reqMktData(contract, snapshot=True)
            await asyncio.sleep(2)

            price = ticker.marketPrice()
            self.conn.ib.cancelMktData(contract)

            if price and price == price:  # NaN check
                return float(price)

            if ticker.last and ticker.last == ticker.last:
                return float(ticker.last)
            if ticker.close and ticker.close == ticker.close:
                return float(ticker.close)

            return None
        except Exception as e:
            self.logger.error(f"Error getting price for {contract.symbol}: {e}")
            return None

    async def get_positions(self) -> list[dict]:
        if not self.conn.connected:
            return []

        positions = self.conn.ib.positions()
        result = []
        for pos in positions:
            result.append({
                "instrument": pos.contract.symbol,
                "side": "LONG" if pos.position > 0 else "SHORT",
                "size": abs(pos.position),
                "entry_price": pos.avgCost,
                "unrealized_pnl": 0.0,
            })
        return result
