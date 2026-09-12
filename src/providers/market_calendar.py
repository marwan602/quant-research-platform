from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo
import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
)

NYSE_TIMEZONE = ZoneInfo("America/New_York")
MARKET_CLOSE_TIME = time(16, 0)


class USTradingCalendar(AbstractHolidayCalendar):
    """
    Authoritative NYSE trading holiday calendar.
    Covers the 10 official annual exchange holidays:
    - New Year's Day (observed nearest workday)
    - Martin Luther King, Jr. Day (3rd Monday of January)
    - Washington's Birthday / Presidents' Day (3rd Monday of February)
    - Good Friday (Friday before Easter Sunday)
    - Memorial Day (last Monday of May)
    - Juneteenth National Independence Day (June 19, established 2021)
    - Independence Day (July 4, observed nearest workday)
    - Labor Day (1st Monday of September)
    - Thanksgiving Day (4th Thursday of November)
    - Christmas Day (December 25, observed nearest workday)
    """

    rules = [
        Holiday("NewYearsDay", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2021-06-18", observance=nearest_workday),
        Holiday("IndependenceDay", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


_CALENDAR_INSTANCE = USTradingCalendar()


def get_nyse_holidays(start_date: str | pd.Timestamp, end_date: str | pd.Timestamp) -> set[pd.Timestamp]:
    """Returns a set of normalized pd.Timestamp holidays within [start_date, end_date]."""
    holidays = _CALENDAR_INSTANCE.holidays(start_date, end_date)
    return set(pd.to_datetime(holidays).normalize())


def is_trading_day(date: str | pd.Timestamp | datetime) -> bool:
    """Returns True if the specified date is an active NYSE trading session."""
    dt = pd.to_datetime(date).normalize()
    if dt.weekday() >= 5:
        return False
    holidays = get_nyse_holidays(dt - pd.Timedelta(days=7), dt + pd.Timedelta(days=7))
    return dt not in holidays


def get_trading_days(
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> list[pd.Timestamp]:
    """Returns a list of all active NYSE trading sessions in [start_date, end_date]."""
    start_dt = pd.to_datetime(start_date).normalize()
    end_dt = pd.to_datetime(end_date).normalize()
    if start_dt > end_dt:
        return []

    date_range = pd.date_range(start=start_dt, end=end_dt, freq="D")
    weekdays = [d for d in date_range if d.weekday() < 5]
    if not weekdays:
        return []

    holidays = get_nyse_holidays(start_dt - pd.Timedelta(days=7), end_dt + pd.Timedelta(days=7))
    return [d for d in weekdays if d not in holidays]


def get_latest_completed_trading_session(as_of: datetime | None = None) -> pd.Timestamp:
    """
    Determines the authoritative latest completed US trading session.

    Logic:
    1. Converts reference time to America/New_York (Eastern Time).
       Naive datetimes are assumed UTC (standard for server/runner timestamps).
    2. Compares against regular market close (16:00 ET / 4:00 PM Eastern).
       - If current time is strictly before 16:00 ET, today's session has not completed;
         candidate starts from the previous calendar day.
       - If current time is at or after 16:00 ET, today's session has completed;
         candidate starts from today.
    3. Walks backwards day-by-day skipping weekends and NYSE holidays until
       reaching the latest completed trading session.
    """
    if as_of is None:
        now_ny = datetime.now(NYSE_TIMEZONE)
    elif as_of.tzinfo is None:
        now_ny = as_of.replace(tzinfo=ZoneInfo("UTC")).astimezone(NYSE_TIMEZONE)
    else:
        now_ny = as_of.astimezone(NYSE_TIMEZONE)

    curr_date = pd.Timestamp(now_ny.date()).normalize()

    # If before market close, today's session is not yet completed
    if now_ny.time() < MARKET_CLOSE_TIME:
        curr_date -= pd.Timedelta(days=1)

    # Search window for holidays
    start_search = curr_date - pd.Timedelta(days=20)
    holidays = get_nyse_holidays(start_search, curr_date + pd.Timedelta(days=1))

    while curr_date.weekday() >= 5 or curr_date in holidays:
        curr_date -= pd.Timedelta(days=1)

    return curr_date
