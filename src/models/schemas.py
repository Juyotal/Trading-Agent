from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class TradeAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class Instrument(BaseModel):
    symbol: str
    exchange: str
    sec_type: str
    currency: str = "USD"
    description: str = ""


class OHLCVBar(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class IndicatorValues(BaseModel):
    ema_9: Optional[float] = None
    ema_21: Optional[float] = None
    ema_50: Optional[float] = None
    rsi_14: Optional[float] = None
    atr_14: Optional[float] = None
    vwap: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    trend: Optional[str] = None


class TradeOutcome(BaseModel):
    instrument: str
    action: str  # "BUY" or "SELL"
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    pnl: float
    exit_reason: str  # "STOP", "TARGET", "MANUAL", "END_OF_DATA"
    reasoning: str = ""


class SessionContext(BaseModel):
    first_bar_of_session: bool = False
    near_session_close: bool = False
    near_weekend_close: bool = False
    minutes_until_close: int = 0


class GapInfo(BaseModel):
    size: float = 0.0
    ratio: float = 0.0
    anomalous: bool = False


class MarketSnapshot(BaseModel):
    instrument: str
    timeframe: str
    timestamp: datetime
    current_price: float
    bars: list[OHLCVBar]
    indicators: IndicatorValues
    current_positions: list[PositionInfo] = Field(default_factory=list)
    account_balance: float = 0.0
    daily_pnl: float = 0.0
    session: Optional[SessionContext] = None
    gap: Optional[GapInfo] = None
    recent_trades: list[TradeOutcome] = Field(default_factory=list)


class PositionInfo(BaseModel):
    instrument: str
    side: str
    size: float
    entry_price: float
    unrealized_pnl: float = 0.0


class TradeDecision(BaseModel):
    action: TradeAction
    instrument: str
    confidence: float = Field(ge=0.0, le=1.0)
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    size: int = 1
    reasoning: str = ""

    @field_validator("confidence")
    @classmethod
    def round_confidence(cls, v: float) -> float:
        return round(v, 3)

    @field_validator("stop_loss")
    @classmethod
    def stop_loss_required_for_trades(cls, v, info):
        action = info.data.get("action")
        if action in (TradeAction.BUY, TradeAction.SELL) and v is None:
            raise ValueError("stop_loss is required for BUY/SELL actions")
        return v

    @field_validator("take_profit")
    @classmethod
    def take_profit_required_for_trades(cls, v, info):
        action = info.data.get("action")
        if action in (TradeAction.BUY, TradeAction.SELL) and v is None:
            raise ValueError("take_profit is required for BUY/SELL actions")
        return v


MarketSnapshot.model_rebuild()
