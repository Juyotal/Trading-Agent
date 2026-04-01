from datetime import datetime

import pytest
import pytz

from src.utils.scheduler import MarketScheduler


@pytest.fixture
def scheduler():
    return MarketScheduler(
        timezone="US/Eastern",
        futures_start="18:00",
        futures_end="17:00",
    )


def freeze(scheduler: MarketScheduler, year, month, day, hour, minute):
    """Pin MarketScheduler._now to a specific ET datetime."""
    fixed = scheduler.tz.localize(datetime(year, month, day, hour, minute))
    scheduler._now = lambda: fixed


class TestMinutesUntilFuturesClose:
    def test_mid_afternoon_thursday(self, scheduler):
        # Thursday 2026-04-16 at 14:00 ET — 3h until 17:00
        freeze(scheduler, 2026, 4, 16, 14, 0)
        assert scheduler.minutes_until_futures_close() == 180

    def test_just_after_close_rolls_to_next_day(self, scheduler):
        # Thursday 17:30 — next 17:00 is Friday, ~23.5h away
        freeze(scheduler, 2026, 4, 16, 17, 30)
        mins = scheduler.minutes_until_futures_close()
        assert 1400 <= mins <= 1420

    def test_one_minute_before_close(self, scheduler):
        freeze(scheduler, 2026, 4, 16, 16, 59)
        assert scheduler.minutes_until_futures_close() == 1


class TestIsNearSessionClose:
    def test_far_from_close(self, scheduler):
        freeze(scheduler, 2026, 4, 16, 10, 0)
        assert scheduler.is_near_session_close(buffer_minutes=15) is False

    def test_within_buffer(self, scheduler):
        freeze(scheduler, 2026, 4, 16, 16, 50)
        assert scheduler.is_near_session_close(buffer_minutes=15) is True

    def test_exactly_at_buffer(self, scheduler):
        freeze(scheduler, 2026, 4, 16, 16, 45)
        assert scheduler.is_near_session_close(buffer_minutes=15) is True

    def test_during_maintenance_break_returns_false(self, scheduler):
        # 17:30 ET is inside the daily break — session is closed, not "near close"
        freeze(scheduler, 2026, 4, 16, 17, 30)
        assert scheduler.is_near_session_close(buffer_minutes=15) is False


class TestIsNearWeekendClose:
    def test_friday_within_buffer(self, scheduler):
        # Friday 2026-04-17 at 16:45 ET — 15m before weekly close
        freeze(scheduler, 2026, 4, 17, 16, 45)
        assert scheduler.is_near_weekend_close(buffer_minutes=30) is True

    def test_friday_far_from_close(self, scheduler):
        freeze(scheduler, 2026, 4, 17, 10, 0)
        assert scheduler.is_near_weekend_close(buffer_minutes=30) is False

    def test_friday_after_close(self, scheduler):
        # After 17:00 Friday — close already happened, not "near"
        freeze(scheduler, 2026, 4, 17, 17, 30)
        assert scheduler.is_near_weekend_close(buffer_minutes=30) is False

    def test_thursday_near_daily_close_not_weekend(self, scheduler):
        freeze(scheduler, 2026, 4, 16, 16, 45)
        assert scheduler.is_near_weekend_close(buffer_minutes=30) is False


class TestIsFirstBarOfSession:
    def test_just_after_open(self, scheduler):
        # 18:05 ET — 5 min into new session
        freeze(scheduler, 2026, 4, 16, 18, 5)
        assert scheduler.is_first_bar_of_session(window_minutes=10) is True

    def test_exactly_at_open(self, scheduler):
        freeze(scheduler, 2026, 4, 16, 18, 0)
        assert scheduler.is_first_bar_of_session(window_minutes=10) is True

    def test_past_window(self, scheduler):
        freeze(scheduler, 2026, 4, 16, 18, 15)
        assert scheduler.is_first_bar_of_session(window_minutes=10) is False

    def test_midday(self, scheduler):
        freeze(scheduler, 2026, 4, 16, 12, 0)
        assert scheduler.is_first_bar_of_session(window_minutes=10) is False
