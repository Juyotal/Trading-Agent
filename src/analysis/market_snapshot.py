from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.analysis.indicators import compute_all_indicators, detect_gap
from src.models.schemas import (
    GapInfo,
    MarketSnapshot,
    OHLCVBar,
    PositionInfo,
    SessionContext,
    TradeOutcome,
)
from src.utils.logger import get_logger


def build_market_snapshot(
    instrument: str,
    timeframe: str,
    df: pd.DataFrame,
    current_price: float,
    positions: list[dict] = None,
    account_balance: float = 0.0,
    daily_pnl: float = 0.0,
    ema_periods: list[int] = None,
    rsi_period: int = 14,
    atr_period: int = 14,
    bb_period: int = 20,
    bb_std: float = 2.0,
    session_ctx: SessionContext = None,
    recent_trades: list[TradeOutcome] = None,
) -> MarketSnapshot:
    logger = get_logger()

    indicators = compute_all_indicators(
        df,
        ema_periods=ema_periods,
        rsi_period=rsi_period,
        atr_period=atr_period,
        bb_period=bb_period,
        bb_std=bb_std,
    )

    recent_bars = df.tail(20)
    bars = [
        OHLCVBar(
            timestamp=row["timestamp"],
            open=round(row["open"], 4),
            high=round(row["high"], 4),
            low=round(row["low"], 4),
            close=round(row["close"], 4),
            volume=round(row["volume"], 2),
        )
        for _, row in recent_bars.iterrows()
    ]

    pos_list = []
    if positions:
        for p in positions:
            pos_list.append(PositionInfo(
                instrument=p["instrument"],
                side=p["side"],
                size=p["size"],
                entry_price=p["entry_price"],
                unrealized_pnl=p.get("unrealized_pnl", 0.0),
            ))

    gap_dict = detect_gap(df, indicators.atr_14)
    gap = GapInfo(**gap_dict)

    snapshot = MarketSnapshot(
        instrument=instrument,
        timeframe=timeframe,
        timestamp=datetime.utcnow(),
        current_price=current_price,
        bars=bars,
        indicators=indicators,
        current_positions=pos_list,
        account_balance=account_balance,
        daily_pnl=daily_pnl,
        session=session_ctx,
        gap=gap,
        recent_trades=recent_trades or [],
    )

    logger.info(
        f"Snapshot built for {instrument}: price={current_price}, "
        f"trend={indicators.trend}, RSI={indicators.rsi_14}"
    )
    return snapshot
