from __future__ import annotations

from typing import Callable

import pandas as pd
import pytest

from src.data_layer import fetcher


@pytest.fixture
def january_aapl() -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-03", "2023-01-31", tz="UTC")
    # Martin Luther King Jr. Day is a market holiday, not a normal business day.
    dates = dates[dates != pd.Timestamp("2023-01-16", tz="UTC")]
    return pd.DataFrame(
        {
            "Open": range(100, 100 + len(dates)),
            "High": range(101, 101 + len(dates)),
            "Low": range(99, 99 + len(dates)),
            "Close": range(100, 100 + len(dates)),
            "Volume": [1_000_000] * len(dates),
        },
        index=dates,
    )


def _downloader_for(
    source: pd.DataFrame, calls: list[dict[str, object]]
) -> Callable[..., pd.DataFrame]:
    def download(_: str, **kwargs: object) -> pd.DataFrame:
        calls.append(kwargs)
        start = pd.Timestamp(str(kwargs["start"]), tz="UTC")
        # yfinance receives an exclusive end date.
        end_exclusive = pd.Timestamp(str(kwargs["end"]), tz="UTC")
        return source.loc[(source.index >= start) & (source.index < end_exclusive)].copy()

    return download


def test_fetches_small_range_with_expected_trading_days(
    monkeypatch: pytest.MonkeyPatch, tmp_path, january_aapl: pd.DataFrame
) -> None:
    monkeypatch.chdir(tmp_path)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(fetcher.yf, "download", _downloader_for(january_aapl, calls))

    result = fetcher.fetch_ohlcv("AAPL", "2023-01-01", "2023-01-31")

    assert len(result) == 20
    assert list(result.columns) == ["open", "high", "low", "close", "volume"]
    assert result.index.name == "date"
    assert result.index.tz is not None
    assert str(result.index.tz) == "UTC"
    assert result.index.is_monotonic_increasing
    assert not result.index.has_duplicates
    assert len(calls) == 1


def test_same_request_uses_cache_without_a_second_network_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path, january_aapl: pd.DataFrame
) -> None:
    monkeypatch.chdir(tmp_path)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(fetcher.yf, "download", _downloader_for(january_aapl, calls))

    fetcher.fetch_ohlcv("AAPL", "2023-01-01", "2023-01-31")
    result = fetcher.fetch_ohlcv("AAPL", "2023-01-01", "2023-01-31")

    assert len(calls) == 1
    assert len(result) == 20


def test_wider_second_request_fetches_only_the_missing_delta(
    monkeypatch: pytest.MonkeyPatch, tmp_path, january_aapl: pd.DataFrame
) -> None:
    monkeypatch.chdir(tmp_path)
    february_dates = pd.bdate_range("2023-02-01", "2023-02-10", tz="UTC")
    full_source = pd.concat(
        [
            january_aapl,
            pd.DataFrame(
                {
                    "Open": range(200, 200 + len(february_dates)),
                    "High": range(201, 201 + len(february_dates)),
                    "Low": range(199, 199 + len(february_dates)),
                    "Close": range(200, 200 + len(february_dates)),
                    "Volume": [1_000_000] * len(february_dates),
                },
                index=february_dates,
            ),
        ]
    )
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(fetcher.yf, "download", _downloader_for(full_source, calls))

    fetcher.fetch_ohlcv("AAPL", "2023-01-01", "2023-01-31")
    result = fetcher.fetch_ohlcv("AAPL", "2023-01-01", "2023-02-10")

    assert len(calls) == 2
    assert calls[1]["start"] == "2023-02-01"
    assert calls[1]["end"] == "2023-02-11"
    assert len(result) == 28


def test_duplicate_timestamp_in_download_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path, january_aapl: pd.DataFrame
) -> None:
    monkeypatch.chdir(tmp_path)
    duplicate = pd.concat([january_aapl, january_aapl.iloc[[0]]])
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(fetcher.yf, "download", _downloader_for(duplicate, calls))

    with pytest.raises(ValueError, match="Duplicate timestamps"):
        fetcher.fetch_ohlcv("AAPL", "2023-01-01", "2023-01-31")


def test_nan_row_is_retained_and_reported(
    monkeypatch: pytest.MonkeyPatch, tmp_path, january_aapl: pd.DataFrame
) -> None:
    monkeypatch.chdir(tmp_path)
    source = january_aapl.copy()
    flagged_date = source.index[4]
    source.loc[flagged_date, "Close"] = float("nan")
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(fetcher.yf, "download", _downloader_for(source, calls))

    result = fetcher.fetch_ohlcv("AAPL", "2023-01-01", "2023-01-31")
    report = result.attrs["data_quality_report"]

    assert pd.isna(result.loc[flagged_date, "close"])
    assert report.flagged_row_count == 1
    assert flagged_date in report.reasons["nan_ohlc"]

def test_expected_sessions_never_includes_today_or_later(
    monkeypatch: pytest.MonkeyPatch, tmp_path, january_aapl: pd.DataFrame
) -> None:
    """A request whose range extends through "today" must never expect a
    session for today or later, since today's bar may not exist yet or be
    final. Requesting the same range twice in a row (simulating two runs on
    the same day) must not produce a duplicate-timestamp error from trying
    to re-fetch an always-missing "today" session."""
    monkeypatch.chdir(tmp_path)
    today = pd.Timestamp.now(tz="UTC").normalize()
    yesterday = today - pd.Timedelta(days=1)

    # Source data only goes up through yesterday -- nothing exists for
    # "today", matching real yfinance behavior for an in-progress session.
    source_dates = pd.bdate_range(yesterday - pd.Timedelta(days=10), yesterday, tz="UTC")
    source = pd.DataFrame(
        {
            "Open": range(100, 100 + len(source_dates)),
            "High": range(101, 101 + len(source_dates)),
            "Low": range(99, 99 + len(source_dates)),
            "Close": range(100, 100 + len(source_dates)),
            "Volume": [1_000_000] * len(source_dates),
        },
        index=source_dates,
    )
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(fetcher.yf, "download", _downloader_for(source, calls))

    start = (yesterday - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    end = today.strftime("%Y-%m-%d")

    # First run: fetches and caches through yesterday.
    first_result = fetcher.fetch_ohlcv("AAPL", start, end)
    # Second run on the "same day": must not re-request a never-available
    # "today" session, and must not raise a duplicate-timestamp error.
    second_result = fetcher.fetch_ohlcv("AAPL", start, end)

    assert len(second_result) == len(first_result)
    assert not second_result.index.has_duplicates
    assert today not in second_result.index