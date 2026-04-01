import pandas as pd
import pytest

from src.analysis.indicators import detect_gap


def make_df(prior_close: float, current_open: float, extra_rows: int = 0) -> pd.DataFrame:
    rows = []
    for _ in range(extra_rows):
        rows.append({"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000})
    rows.append({
        "open": prior_close,
        "high": prior_close + 1,
        "low": prior_close - 1,
        "close": prior_close,
        "volume": 1000,
    })
    rows.append({
        "open": current_open,
        "high": current_open + 1,
        "low": current_open - 1,
        "close": current_open,
        "volume": 1000,
    })
    return pd.DataFrame(rows)


class TestDetectGap:
    def test_no_gap(self):
        df = make_df(prior_close=5000.0, current_open=5000.0)
        result = detect_gap(df, atr=10.0)
        assert result["size"] == 0.0
        assert result["ratio"] == 0.0
        assert result["anomalous"] is False

    def test_small_gap_not_anomalous(self):
        df = make_df(prior_close=5000.0, current_open=5005.0)
        result = detect_gap(df, atr=10.0)
        assert result["size"] == 5.0
        assert result["ratio"] == 0.5
        assert result["anomalous"] is False

    def test_large_gap_anomalous(self):
        df = make_df(prior_close=5000.0, current_open=5030.0)
        result = detect_gap(df, atr=10.0)
        assert result["size"] == 30.0
        assert result["ratio"] == 3.0
        assert result["anomalous"] is True

    def test_at_threshold_is_anomalous(self):
        df = make_df(prior_close=5000.0, current_open=5020.0)
        result = detect_gap(df, atr=10.0)
        assert result["ratio"] == 2.0
        assert result["anomalous"] is True

    def test_negative_gap_uses_absolute_ratio(self):
        df = make_df(prior_close=5000.0, current_open=4970.0)
        result = detect_gap(df, atr=10.0)
        assert result["size"] == -30.0
        assert result["ratio"] == 3.0
        assert result["anomalous"] is True

    def test_insufficient_bars(self):
        df = pd.DataFrame([{"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1}])
        result = detect_gap(df, atr=10.0)
        assert result == {"size": 0.0, "ratio": 0.0, "anomalous": False}

    def test_none_atr(self):
        df = make_df(prior_close=5000.0, current_open=5050.0)
        result = detect_gap(df, atr=None)
        assert result["size"] == 50.0
        assert result["ratio"] == 0.0
        assert result["anomalous"] is False

    def test_zero_atr(self):
        df = make_df(prior_close=5000.0, current_open=5050.0)
        result = detect_gap(df, atr=0.0)
        assert result["ratio"] == 0.0
        assert result["anomalous"] is False
