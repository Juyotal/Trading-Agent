from __future__ import annotations

from datetime import datetime, time, timedelta

import pytz

from src.utils.logger import get_logger


class MarketScheduler:
    """Determines whether the market is open and manages analysis intervals."""

    def __init__(
        self,
        timezone: str = "US/Eastern",
        futures_start: str = "18:00",
        futures_end: str = "17:00",
    ):
        self.tz = pytz.timezone(timezone)
        self.futures_start = self._parse_time(futures_start)
        self.futures_end = self._parse_time(futures_end)
        self.logger = get_logger()

    def _now(self) -> datetime:
        return datetime.now(self.tz)

    @staticmethod
    def _parse_time(time_str: str) -> time:
        parts = time_str.split(":")
        return time(int(parts[0]), int(parts[1]))

    def is_futures_session(self) -> bool:
        """CME futures trade Sun 6PM - Fri 5PM ET with a daily break 5PM-6PM."""
        now = self._now()
        current_time = now.time()
        weekday = now.weekday()

        # Sat (5) or Sun before 6PM (6)
        if weekday == 5:
            return False
        if weekday == 6 and current_time < self.futures_start:
            return False

        # Daily maintenance break: 5PM - 6PM ET
        if self.futures_end <= current_time < self.futures_start:
            return False

        return True

    def is_forex_session(self) -> bool:
        """Forex trades Sun 5PM - Fri 5PM ET."""
        now = self._now()
        weekday = now.weekday()
        current_time = now.time()

        if weekday == 5:
            return False
        if weekday == 6 and current_time < time(17, 0):
            return False

        return True

    def is_market_open(self, sec_type: str) -> bool:
        if sec_type == "FUT":
            return self.is_futures_session()
        elif sec_type == "CASH":
            return self.is_forex_session()
        return False

    def is_high_activity_window(self) -> bool:
        """US regular hours 9:30 AM - 4:00 PM ET — best for intraday trading."""
        now = self._now()
        current_time = now.time()
        return time(9, 30) <= current_time <= time(16, 0)

    def minutes_until_futures_close(self) -> int:
        """Minutes until the next 17:00 ET session break (or weekend close)."""
        now = self._now()
        close_today = now.replace(
            hour=self.futures_end.hour,
            minute=self.futures_end.minute,
            second=0,
            microsecond=0,
        )
        if now >= close_today:
            close_today += timedelta(days=1)
        delta = close_today - now
        return int(delta.total_seconds() // 60)

    def is_near_session_close(self, buffer_minutes: int = 15) -> bool:
        """True when within buffer_minutes of the daily 17:00 ET close."""
        if not self.is_futures_session():
            return False
        return self.minutes_until_futures_close() <= buffer_minutes

    def is_near_weekend_close(self, buffer_minutes: int = 30) -> bool:
        """True on Friday within buffer_minutes of the 17:00 ET weekly close."""
        now = self._now()
        if now.weekday() != 4:  # Friday
            return False
        close = now.replace(
            hour=self.futures_end.hour,
            minute=self.futures_end.minute,
            second=0,
            microsecond=0,
        )
        if now >= close:
            return False
        delta_min = int((close - now).total_seconds() // 60)
        return delta_min <= buffer_minutes

    def is_first_bar_of_session(self, window_minutes: int = 10) -> bool:
        """True during the first window_minutes of a new futures session (post 18:00 ET)."""
        now = self._now()
        current = now.time()
        # Window spans futures_start to futures_start + window_minutes
        start = self.futures_start
        start_minutes = start.hour * 60 + start.minute
        cur_minutes = current.hour * 60 + current.minute
        delta = cur_minutes - start_minutes
        if delta < 0:
            delta += 24 * 60
        return 0 <= delta < window_minutes

    def get_session_info(self) -> dict:
        now = self._now()
        return {
            "current_time": now.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "weekday": now.strftime("%A"),
            "futures_open": self.is_futures_session(),
            "forex_open": self.is_forex_session(),
            "high_activity": self.is_high_activity_window(),
            "minutes_until_close": self.minutes_until_futures_close(),
            "near_session_close": self.is_near_session_close(),
            "near_weekend_close": self.is_near_weekend_close(),
            "first_bar_of_session": self.is_first_bar_of_session(),
        }
