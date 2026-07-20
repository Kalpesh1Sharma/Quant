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
