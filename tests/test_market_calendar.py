from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
import pytest

from src.providers.market_calendar import (
    NYSE_TIMEZONE,
    get_latest_completed_trading_session,
    get_nyse_holidays,
    get_trading_days,
    is_trading_day,
)


def test_nyse_holidays_2026():
    holidays = get_nyse_holidays("2026-01-01", "2026-12-31")
    expected_dates = {
        pd.Timestamp("2026-01-01"),  # New Year's Day
        pd.Timestamp("2026-01-19"),  # Martin Luther King, Jr. Day
        pd.Timestamp("2026-02-16"),  # Presidents' Day
        pd.Timestamp("2026-04-03"),  # Good Friday
        pd.Timestamp("2026-05-25"),  # Memorial Day
        pd.Timestamp("2026-06-19"),  # Juneteenth
        pd.Timestamp("2026-07-03"),  # Independence Day (observed)
        pd.Timestamp("2026-09-07"),  # Labor Day
        pd.Timestamp("2026-11-26"),  # Thanksgiving Day
        pd.Timestamp("2026-12-25"),  # Christmas Day
    }
    assert expected_dates.issubset(holidays)


def test_is_trading_day():
    # Regular trading days
    assert is_trading_day("2026-09-11") is True   # Friday
    assert is_trading_day("2026-09-10") is True   # Thursday
    assert is_trading_day("2026-09-08") is True   # Tuesday

    # Weekends
    assert is_trading_day("2026-09-12") is False  # Saturday
    assert is_trading_day("2026-09-13") is False  # Sunday

    # NYSE Holidays
    assert is_trading_day("2026-09-07") is False  # Labor Day
    assert is_trading_day("2026-04-03") is False  # Good Friday
    assert is_trading_day("2026-06-19") is False  # Juneteenth


def test_get_trading_days_skips_weekends_and_holidays():
    # Span includes Friday Sep 4, Weekend (Sep 5-6), Labor Day (Sep 7), and Tue-Fri (Sep 8-11)
    days = get_trading_days("2026-09-04", "2026-09-11")
    expected = [
        pd.Timestamp("2026-09-04"),
        pd.Timestamp("2026-09-08"),
        pd.Timestamp("2026-09-09"),
        pd.Timestamp("2026-09-10"),
        pd.Timestamp("2026-09-11"),
    ]
    assert days == expected


def test_get_latest_completed_trading_session_scenarios():
    # 1. Friday during trading hours (15:30 NY) -> Thursday
    dt1 = datetime(2026, 9, 11, 15, 30, tzinfo=NYSE_TIMEZONE)
    assert get_latest_completed_trading_session(dt1) == pd.Timestamp("2026-09-10")

    # 2. Friday after market close (16:05 NY) -> Friday
    dt2 = datetime(2026, 9, 11, 16, 5, tzinfo=NYSE_TIMEZONE)
    assert get_latest_completed_trading_session(dt2) == pd.Timestamp("2026-09-11")

    # 3. Friday 22:00 UTC (18:00 NY, runner scheduled time) -> Friday
    dt3 = datetime(2026, 9, 11, 22, 0, tzinfo=ZoneInfo("UTC"))
    assert get_latest_completed_trading_session(dt3) == pd.Timestamp("2026-09-11")

    # 4. Saturday after midnight UTC (00:30 UTC = Friday 20:30 NY) -> Friday
    dt4 = datetime(2026, 9, 12, 0, 30, tzinfo=ZoneInfo("UTC"))
    assert get_latest_completed_trading_session(dt4) == pd.Timestamp("2026-09-11")

    # 5. Saturday midday (14:00 NY) -> Friday
    dt5 = datetime(2026, 9, 12, 14, 0, tzinfo=NYSE_TIMEZONE)
    assert get_latest_completed_trading_session(dt5) == pd.Timestamp("2026-09-11")

    # 6. Sunday night (23:00 NY) -> Friday
    dt6 = datetime(2026, 9, 13, 23, 0, tzinfo=NYSE_TIMEZONE)
    assert get_latest_completed_trading_session(dt6) == pd.Timestamp("2026-09-11")

    # 7. Monday pre-market (08:30 NY) -> Friday
    dt7 = datetime(2026, 9, 14, 8, 30, tzinfo=NYSE_TIMEZONE)
    assert get_latest_completed_trading_session(dt7) == pd.Timestamp("2026-09-11")

    # 8. Monday after close (16:30 NY) -> Monday
    dt8 = datetime(2026, 9, 14, 16, 30, tzinfo=NYSE_TIMEZONE)
    assert get_latest_completed_trading_session(dt8) == pd.Timestamp("2026-09-14")

    # 9. Tuesday morning after Labor Day (2026-09-08 09:00 NY) -> Friday Sep 4
    dt9 = datetime(2026, 9, 8, 9, 0, tzinfo=NYSE_TIMEZONE)
    assert get_latest_completed_trading_session(dt9) == pd.Timestamp("2026-09-04")

    # 10. Naive datetime assumes UTC and converts correctly
    dt10 = datetime(2026, 9, 11, 22, 0)  # Naive 22:00 UTC
    assert get_latest_completed_trading_session(dt10) == pd.Timestamp("2026-09-11")
