from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.models.schemas import IndicatorValues
from src.utils.logger import get_logger

logger = get_logger()


def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / period, min_periods=period).mean()


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cumulative_tp_vol = (typical_price * df["volume"]).cumsum()
    cumulative_vol = df["volume"].cumsum()
    return cumulative_tp_vol / cumulative_vol.replace(0, np.nan)


def compute_bollinger_bands(
    series: pd.Series, period: int = 20, std_dev: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    middle = series.rolling(window=period).mean()
    rolling_std = series.rolling(window=period).std()
    upper = middle + (std_dev * rolling_std)
    lower = middle - (std_dev * rolling_std)
    return upper, middle, lower


def detect_trend(df: pd.DataFrame, ema_short: pd.Series, ema_long: pd.Series) -> str:
    if ema_short.iloc[-1] > ema_long.iloc[-1]:
        if df["close"].iloc[-1] > ema_short.iloc[-1]:
            return "STRONG_UPTREND"
        return "UPTREND"
    elif ema_short.iloc[-1] < ema_long.iloc[-1]:
        if df["close"].iloc[-1] < ema_short.iloc[-1]:
            return "STRONG_DOWNTREND"
        return "DOWNTREND"
    return "SIDEWAYS"


def detect_gap(df: pd.DataFrame, atr: Optional[float]) -> dict:
    """Detect a price gap between the most recent bar and the one before it.

    Returns size (signed price delta), ratio (size / atr), and anomalous flag.
    """
    if len(df) < 2:
        return {"size": 0.0, "ratio": 0.0, "anomalous": False}

    prior_close = float(df["close"].iloc[-2])
    current_open = float(df["open"].iloc[-1])
    gap_size = current_open - prior_close

    if atr is None or atr <= 0:
        ratio = 0.0
    else:
        ratio = abs(gap_size) / atr

    return {
        "size": round(gap_size, 4),
        "ratio": round(ratio, 3),
        "anomalous": ratio >= 2.0,
    }


def compute_all_indicators(
    df: pd.DataFrame,
    ema_periods: list[int] = None,
    rsi_period: int = 14,
    atr_period: int = 14,
    bb_period: int = 20,
    bb_std: float = 2.0,
) -> IndicatorValues:
    if ema_periods is None:
        ema_periods = [9, 21, 50]

    if len(df) < max(ema_periods + [rsi_period, atr_period, bb_period]):
        logger.warning("Not enough bars for full indicator computation")

    close = df["close"]

    ema_values = {}
    for period in ema_periods:
        ema = compute_ema(close, period)
        ema_values[period] = ema

    rsi = compute_rsi(close, rsi_period)
    atr = compute_atr(df, atr_period)
    vwap = compute_vwap(df)
    bb_upper, bb_middle, bb_lower = compute_bollinger_bands(close, bb_period, bb_std)

    ema_9 = ema_values.get(9)
    ema_21 = ema_values.get(21)
    ema_50 = ema_values.get(50)

    trend = detect_trend(
        df,
        ema_9 if ema_9 is not None else compute_ema(close, 9),
        ema_50 if ema_50 is not None else compute_ema(close, 50),
    )

    def safe_last(series: Optional[pd.Series]) -> Optional[float]:
        if series is None or series.empty:
            return None
        val = series.iloc[-1]
        return round(float(val), 4) if pd.notna(val) else None

    return IndicatorValues(
        ema_9=safe_last(ema_9),
        ema_21=safe_last(ema_21),
        ema_50=safe_last(ema_50),
        rsi_14=safe_last(rsi),
        atr_14=safe_last(atr),
        vwap=safe_last(vwap),
        bb_upper=safe_last(bb_upper),
        bb_middle=safe_last(bb_middle),
        bb_lower=safe_last(bb_lower),
        trend=trend,
    )
