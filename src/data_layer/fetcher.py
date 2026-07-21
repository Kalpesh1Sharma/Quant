"""Fetch, cache, and validate OHLCV market data.

The public functions in this module deliberately have no knowledge of a
strategy or backtest.  Quality findings are available on every returned
``DataFrame`` through ``frame.attrs["data_quality_report"]``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd
import yfinance as yf
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
from pandas.tseries.offsets import CustomBusinessDay


LOGGER = logging.getLogger(__name__)
REQUIRED_COLUMNS: Final[tuple[str, ...]] = ("open", "high", "low", "close", "volume")
OHLC_COLUMNS: Final[tuple[str, ...]] = ("open", "high", "low", "close")


class _NYSEHolidayCalendar(AbstractHolidayCalendar):
    """Fallback for the regular NYSE holidays when the calendar package is absent."""

    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday(
            "Juneteenth National Independence Day",
            month=6,
            day=19,
            start_date="2022-06-19",
            observance=nearest_workday,
        ),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas Day", month=12, day=25, observance=nearest_workday),
    ]


@dataclass(frozen=True)
class DataQualityReport:
    """Non-destructive data-quality findings associated with a returned frame.

    ``flagged_row_count`` covers rows with zero volume or a NaN OHLC value.
    ``reasons`` maps each finding type to its affected UTC dates; it includes
    ``unexplained_gap`` dates, which are absent from the frame and therefore
    are not counted as flagged rows.
    """

    flagged_row_count: int
    flagged_dates: tuple[pd.Timestamp, ...]
    reasons: dict[str, tuple[pd.Timestamp, ...]]


def fetch_ohlcv(ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    """Fetch a requested inclusive date range and maintain its Parquet cache.

    Cached files live at ``data/raw/{ticker}_{interval}.parquet`` relative to
    the current working directory.  The returned frame has a UTC ``date``
    index and exposes a :class:`DataQualityReport` in
    ``frame.attrs["data_quality_report"]``.
    """

    safe_ticker = _validate_ticker(ticker)
    request_start, request_end = _parse_requested_range(start, end, interval)
    cached = _read_cache(safe_ticker, interval)

    if cached is None:
        merged = _download(safe_ticker, request_start, request_end, interval)
        _assert_no_duplicate_dates(merged, safe_ticker)
        _write_cache(safe_ticker, interval, merged)
    else:
        missing_ranges = _missing_ranges(cached, request_start, request_end, interval)
        if missing_ranges:
            deltas = [
                _download(safe_ticker, missing_start, missing_end, interval)
                for missing_start, missing_end in missing_ranges
            ]
            merged = pd.concat([cached, *deltas])
            _assert_no_duplicate_dates(merged, safe_ticker)
            merged = merged.sort_index()
            _write_cache(safe_ticker, interval, merged)
        else:
            merged = cached

    result = _slice_to_request(merged, request_start, request_end, interval)
    return _finalise(result, safe_ticker, request_start, request_end, interval)


def load_cached(ticker: str, interval: str = "1d") -> pd.DataFrame | None:
    """Return the full local cache for ``ticker``/``interval`` without I/O to Yahoo."""

    safe_ticker = _validate_ticker(ticker)
    cached = _read_cache(safe_ticker, interval)
    if cached is None:
        return None

    quality_start = cached.index.min() if not cached.empty else None
    quality_end = cached.index.max() if not cached.empty else None
    return _finalise(cached, safe_ticker, quality_start, quality_end, interval)


def _cache_path(ticker: str, interval: str) -> Path:
    return Path("data") / "raw" / f"{ticker}_{interval}.parquet"


def _validate_ticker(ticker: str) -> str:
    cleaned = ticker.strip()
    if not cleaned:
        raise ValueError("ticker must not be empty")
    if any(separator in cleaned for separator in ("/", "\\")) or cleaned in {".", ".."}:
        raise ValueError(f"ticker must be a filename-safe symbol, got {ticker!r}")
    return cleaned


def _parse_requested_range(start: str, end: str, interval: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    try:
        start_timestamp = pd.Timestamp(start)
        end_timestamp = pd.Timestamp(end)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid requested range: start={start!r}, end={end!r}") from error

    start_utc = _as_utc(start_timestamp)
    end_utc = _as_utc(end_timestamp)
    if _is_daily_interval(interval):
        start_utc = start_utc.normalize()
        end_utc = end_utc.normalize()
    if end_utc < start_utc:
        raise ValueError(f"end must be on or after start, got start={start!r}, end={end!r}")
    return start_utc, end_utc


def _as_utc(value: pd.Timestamp) -> pd.Timestamp:
    return value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")


def _is_daily_interval(interval: str) -> bool:
    return interval.lower() == "1d"


def _read_cache(ticker: str, interval: str) -> pd.DataFrame | None:
    path = _cache_path(ticker, interval)
    if not path.exists():
        return None
    try:
        return _normalise_ohlcv(pd.read_parquet(path), interval)
    except Exception as error:
        raise RuntimeError(f"Could not read OHLCV cache for {ticker!r} at {path}") from error


def _write_cache(ticker: str, interval: str, data: pd.DataFrame) -> None:
    path = _cache_path(ticker, interval)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.stem}.tmp.parquet")
    data.to_parquet(temporary_path)
    temporary_path.replace(path)


def _download(ticker: str, start: pd.Timestamp, end: pd.Timestamp, interval: str) -> pd.DataFrame:
    """Download an inclusive range; yfinance's ``end`` parameter is exclusive."""

    download_end = end + pd.Timedelta(days=1) if _is_daily_interval(interval) else end
    try:
        downloaded = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=download_end.strftime("%Y-%m-%d"),
            interval=interval,
            auto_adjust=False,
            progress=False,
        )
    except Exception as error:
        raise RuntimeError(
            f"Failed to fetch OHLCV for {ticker!r} from {start.isoformat()} to {end.isoformat()} "
            f"at interval {interval!r}"
        ) from error

    try:
        return _normalise_ohlcv(downloaded, interval)
    except Exception as error:
        raise RuntimeError(
            f"Received invalid OHLCV data for {ticker!r} from {start.isoformat()} to {end.isoformat()} "
            f"at interval {interval!r}"
        ) from error


def _normalise_ohlcv(data: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Select required fields and give them a sorted, UTC ``date`` index."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError("yfinance must return a pandas DataFrame")

    frame = data.copy()
    date_column = next((column for column in frame.columns if str(column).lower() == "date"), None)
    if date_column is not None:
        frame = frame.set_index(date_column)
    if isinstance(frame.index, pd.RangeIndex):
        raise ValueError("OHLCV data must have a DatetimeIndex or a date column")

    selected_columns: dict[str, pd.Series] = {}
    for expected in REQUIRED_COLUMNS:
        column = _find_column(frame, expected)
        if column is None:
            raise ValueError(f"OHLCV data is missing required column {expected!r}")
        value = frame.loc[:, column]
        if isinstance(value, pd.DataFrame):
            value = value.iloc[:, 0]
        selected_columns[expected] = value

    normalised = pd.DataFrame(selected_columns, index=frame.index)
    normalised.index = pd.DatetimeIndex(pd.to_datetime(normalised.index, utc=True, errors="raise"))
    if _is_daily_interval(interval):
        normalised.index = normalised.index.normalize()
    normalised.index.name = "date"
    return normalised.sort_index()


def _find_column(frame: pd.DataFrame, expected: str) -> object | None:
    for column in frame.columns:
        labels = column if isinstance(column, tuple) else (column,)
        if any(str(label).lower() == expected for label in labels):
            return column
    return None


def _missing_ranges(
    cached: pd.DataFrame,
    request_start: pd.Timestamp,
    request_end: pd.Timestamp,
    interval: str,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return only the trading-session spans absent from the cache."""

    expected_sessions = _expected_sessions(request_start, request_end)
    if expected_sessions.empty:
        return []

    cached_sessions = pd.DatetimeIndex(cached.index).normalize().unique()
    present = expected_sessions.isin(cached_sessions)
    missing = ~present
    ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    range_start: pd.Timestamp | None = None
    range_end: pd.Timestamp | None = None

    for session, is_missing in zip(expected_sessions, missing, strict=True):
        if is_missing:
            if range_start is None:
                range_start = session
            range_end = session
        elif range_start is not None and range_end is not None:
            ranges.append((range_start, range_end))
            range_start = None
            range_end = None
    if range_start is not None and range_end is not None:
        ranges.append((range_start, range_end))
    return ranges


def _expected_sessions(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """Use the NYSE schedule when installed, with a standard-holiday fallback.

    Caps the effective end at yesterday (UTC), since the current day's bar
    may not yet exist or be final in yfinance's data -- treating an
    in-progress "today" as an expected, retryable session causes the same
    calendar date to be re-fetched on every run, colliding with the already-
    cached row for the prior confirmed day during merge.
    """
    today = pd.Timestamp.now(tz="UTC").normalize()
    effective_end = min(end, today - pd.Timedelta(days=1))
    if effective_end < start:
        return pd.DatetimeIndex([], tz="UTC")

    try:
        import pandas_market_calendars as market_calendars
    except ImportError:
        business_day = CustomBusinessDay(calendar=_NYSEHolidayCalendar())
        return pd.date_range(start.normalize(), effective_end.normalize(), freq=business_day, tz="UTC")

    schedule = market_calendars.get_calendar("NYSE").schedule(
        start_date=start.date(), end_date=effective_end.date()
    )
    return pd.DatetimeIndex(pd.to_datetime(schedule.index, utc=True)).normalize()


def _slice_to_request(
    data: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    interval: str,
) -> pd.DataFrame:
    if data.empty:
        return data.copy()
    return data.loc[(data.index >= start) & (data.index <= end)].copy()


def _assert_no_duplicate_dates(data: pd.DataFrame, ticker: str) -> None:
    duplicate_dates = data.index[data.index.duplicated(keep=False)].unique().sort_values()
    if not duplicate_dates.empty:
        formatted_dates = ", ".join(timestamp.isoformat() for timestamp in duplicate_dates)
        raise ValueError(f"Duplicate timestamps found for {ticker!r}: {formatted_dates}")


def _finalise(
    data: pd.DataFrame,
    ticker: str,
    quality_start: pd.Timestamp | None,
    quality_end: pd.Timestamp | None,
    interval: str,
) -> pd.DataFrame:
    result = _normalise_ohlcv(data, interval)
    _assert_no_duplicate_dates(result, ticker)
    report = _build_quality_report(result, quality_start, quality_end, interval)
    result.attrs["data_quality_report"] = report
    _log_quality_report(ticker, report)
    return result


def _build_quality_report(
    data: pd.DataFrame,
    range_start: pd.Timestamp | None,
    range_end: pd.Timestamp | None,
    interval: str,
) -> DataQualityReport:
    zero_volume_dates = tuple(data.index[data["volume"].eq(0)].unique().sort_values())
    nan_ohlc_dates = tuple(data.index[data.loc[:, OHLC_COLUMNS].isna().any(axis=1)].unique().sort_values())

    unexplained_gap_dates: tuple[pd.Timestamp, ...] = ()
    if _is_daily_interval(interval) and range_start is not None and range_end is not None:
        expected = _expected_sessions(range_start, range_end)
        actual = pd.DatetimeIndex(data.index).normalize().unique()
        unexplained_gap_dates = tuple(expected.difference(actual).sort_values())

    row_flag_dates = pd.DatetimeIndex([*zero_volume_dates, *nan_ohlc_dates]).unique().sort_values()
    all_flagged_dates = pd.DatetimeIndex([*row_flag_dates, *unexplained_gap_dates]).unique().sort_values()
    return DataQualityReport(
        flagged_row_count=len(row_flag_dates),
        flagged_dates=tuple(all_flagged_dates),
        reasons={
            "zero_volume": zero_volume_dates,
            "nan_ohlc": nan_ohlc_dates,
            "unexplained_gap": unexplained_gap_dates,
        },
    )


def _log_quality_report(ticker: str, report: DataQualityReport) -> None:
    if report.reasons["unexplained_gap"]:
        dates = ", ".join(date.strftime("%Y-%m-%d") for date in report.reasons["unexplained_gap"])
        LOGGER.warning("Unexplained OHLCV gaps for %s: %s", ticker, dates)
    if report.flagged_row_count:
        details = "; ".join(
            f"{reason}={', '.join(date.strftime('%Y-%m-%d') for date in dates)}"
            for reason, dates in report.reasons.items()
            if reason != "unexplained_gap" and dates
        )
        LOGGER.warning("OHLCV row-quality flags for %s: %s", ticker, details)