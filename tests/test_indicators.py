import numpy as np
import pandas as pd
import pytest

from src.analysis.indicators import (
    compute_all_indicators,
    compute_atr,
    compute_bollinger_bands,
    compute_ema,
    compute_rsi,
    compute_vwap,
    detect_trend,
)


@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 60
    base_price = 5000.0
    closes = base_price + np.cumsum(np.random.randn(n) * 10)
    highs = closes + np.abs(np.random.randn(n) * 5)
    lows = closes - np.abs(np.random.randn(n) * 5)
    opens = closes + np.random.randn(n) * 3
    volumes = np.random.randint(100, 10000, n).astype(float)
    timestamps = pd.date_range("2025-01-01 09:30", periods=n, freq="5min")

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })


class TestEMA:
    def test_ema_length(self, sample_df):
        ema = compute_ema(sample_df["close"], 9)
        assert len(ema) == len(sample_df)

    def test_ema_converges_to_price(self, sample_df):
        constant = pd.Series([100.0] * 50)
        ema = compute_ema(constant, 9)
        assert abs(ema.iloc[-1] - 100.0) < 0.01

    def test_shorter_ema_more_responsive(self, sample_df):
        ema_9 = compute_ema(sample_df["close"], 9)
        ema_50 = compute_ema(sample_df["close"], 50)
        diff_9 = abs(ema_9.iloc[-1] - sample_df["close"].iloc[-1])
        diff_50 = abs(ema_50.iloc[-1] - sample_df["close"].iloc[-1])
        assert diff_9 <= diff_50


class TestRSI:
    def test_rsi_range(self, sample_df):
        rsi = compute_rsi(sample_df["close"], 14)
        valid = rsi.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_rsi_overbought_on_uptrend(self):
        prices = pd.Series([100 + i * 2 for i in range(30)], dtype=float)
        rsi = compute_rsi(prices, 14)
        assert rsi.iloc[-1] > 70

    def test_rsi_oversold_on_downtrend(self):
        prices = pd.Series([200 - i * 2 for i in range(30)], dtype=float)
        rsi = compute_rsi(prices, 14)
        assert rsi.iloc[-1] < 30


class TestATR:
    def test_atr_positive(self, sample_df):
        atr = compute_atr(sample_df, 14)
        valid = atr.dropna()
        assert (valid > 0).all()

    def test_atr_increases_with_volatility(self):
        n = 50
        calm = pd.DataFrame({
            "high": [101.0] * n,
            "low": [99.0] * n,
            "close": [100.0] * n,
        })
        volatile = pd.DataFrame({
            "high": [110.0] * n,
            "low": [90.0] * n,
            "close": [100.0] * n,
        })
        assert compute_atr(calm, 14).iloc[-1] < compute_atr(volatile, 14).iloc[-1]


class TestVWAP:
    def test_vwap_between_high_low(self, sample_df):
        vwap = compute_vwap(sample_df)
        valid = vwap.dropna()
        assert len(valid) > 0


class TestBollingerBands:
    def test_upper_above_lower(self, sample_df):
        upper, middle, lower = compute_bollinger_bands(sample_df["close"])
        valid_idx = upper.dropna().index
        assert (upper[valid_idx] >= lower[valid_idx]).all()

    def test_middle_is_sma(self, sample_df):
        _, middle, _ = compute_bollinger_bands(sample_df["close"], 20)
        sma = sample_df["close"].rolling(20).mean()
        valid = middle.dropna().index
        pd.testing.assert_series_equal(middle[valid], sma[valid], check_names=False)


class TestTrend:
    def test_uptrend_when_ema_short_above_long(self, sample_df):
        ema_short = pd.Series([105.0] * len(sample_df))
        ema_long = pd.Series([100.0] * len(sample_df))
        df_up = sample_df.copy()
        df_up["close"] = 110.0
        assert detect_trend(df_up, ema_short, ema_long) == "STRONG_UPTREND"

    def test_downtrend_when_ema_short_below_long(self, sample_df):
        ema_short = pd.Series([95.0] * len(sample_df))
        ema_long = pd.Series([100.0] * len(sample_df))
        df_down = sample_df.copy()
        df_down["close"] = 90.0
        assert detect_trend(df_down, ema_short, ema_long) == "STRONG_DOWNTREND"


class TestComputeAllIndicators:
    def test_returns_indicator_values(self, sample_df):
        indicators = compute_all_indicators(sample_df)
        assert indicators.ema_9 is not None
        assert indicators.ema_21 is not None
        assert indicators.rsi_14 is not None
        assert indicators.atr_14 is not None
        assert indicators.trend is not None

    def test_custom_periods(self, sample_df):
        indicators = compute_all_indicators(
            sample_df, ema_periods=[5, 10, 20], rsi_period=7
        )
        assert indicators.trend is not None
