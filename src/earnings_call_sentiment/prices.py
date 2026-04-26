"""Phase C: download prices and compute look-ahead-safe forward returns.

For each (ticker, report_date) in data/extractions/*.json:
  1. Download daily OHLC from yfinance for the ticker (cached to data/prices/raw/).
  2. Set the ANCHOR to the first trading day STRICTLY AFTER report_date. This is
     the look-ahead-safe entry: we assume the call could have occurred after the
     market closed on report_date, so the earliest a strategy could enter is the
     next session.
  3. Look up close prices at +1, +5, +21, +63 trading days from the anchor.
  4. Compute total returns and excess returns vs SPY at each horizon.

Outputs:
  - data/prices/returns.csv  (one row per call)
  - data/prices/_summary.json
  - data/prices/raw/<TICKER>.csv  (cached OHLC, idempotent)

Idempotent: re-running skips already-cached OHLC. --force re-downloads.
"""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
EXTRACTIONS_DIR = REPO_ROOT / "data" / "extractions"
PRICES_DIR = REPO_ROOT / "data" / "prices"
PRICES_RAW_DIR = PRICES_DIR / "raw"
RETURNS_PATH = PRICES_DIR / "returns.csv"
SUMMARY_PATH = PRICES_DIR / "_summary.json"

BENCHMARK = "SPY"
HORIZONS = [1, 5, 21, 63]          # forward trading days from the T+1 anchor
PRE_CALL_HORIZONS = [5, 21, 63]    # backward trading days from the pre-call anchor (T-1)
# 63 trading days ~= 88 calendar days; 95 keeps headroom for holidays and the
# trading day strictly before report_date (the pre-call anchor).
LOOKBACK_BUFFER_DAYS = 95
LOOKAHEAD_BUFFER_DAYS = 120        # calendar days after latest report_date (covers +63 trading days)


def load_calls() -> list[dict[str, Any]]:
    """One dict per extraction: ticker, quarter, year, report_date (date), filename."""
    calls = []
    for p in sorted(EXTRACTIONS_DIR.glob("*.json")):
        if p.name.startswith("_"):
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        rd = d.get("report_date")
        ticker = d.get("ticker")
        if not (rd and ticker):
            continue
        try:
            report_date = datetime.strptime(rd, "%Y-%m-%d").date()
        except (TypeError, ValueError):
            continue
        calls.append({
            "ticker": ticker,
            "quarter": d.get("quarter"),
            "year": d.get("year"),
            "report_date": report_date,
            "filename": p.name,
        })
    return calls


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance sometimes returns multi-level columns; flatten to single 'Close'."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def download_ohlc(ticker: str, start: date, end: date, force: bool) -> pd.DataFrame:
    """Cache-first daily OHLC. Returns DataFrame indexed by Date with at least 'Close'."""
    cache_path = PRICES_RAW_DIR / f"{ticker}.csv"
    if cache_path.exists() and not force:
        df = pd.read_csv(cache_path, parse_dates=["Date"], index_col="Date")
        if not df.empty and df.index.min().date() <= start and df.index.max().date() >= end:
            return df

    df = yf.download(
        ticker,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        progress=False,
        auto_adjust=True,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    df = _normalize_columns(df)
    df.index.name = "Date"
    df.to_csv(cache_path)
    return df


def find_trading_day_after(prices: pd.DataFrame, after: date) -> date | None:
    """First trading day in `prices` strictly after `after`."""
    if prices.empty:
        return None
    mask = prices.index.date > after
    if not mask.any():
        return None
    return prices.index[mask][0].date()


def find_trading_day_before(prices: pd.DataFrame, before: date) -> date | None:
    """Last trading day in `prices` strictly before `before`."""
    if prices.empty:
        return None
    mask = prices.index.date < before
    if not mask.any():
        return None
    return prices.index[mask][-1].date()


def shift_n_trading_days(prices: pd.DataFrame, anchor: date, n: int) -> date | None:
    """Trading day n bars from `anchor` (positive = after, negative = before).

    None if the resulting index is out of range.
    """
    if prices.empty:
        return None
    dates = list(prices.index.date)
    try:
        i = dates.index(anchor)
    except ValueError:
        return None
    target = i + n
    if target < 0 or target >= len(dates):
        return None
    return dates[target]


def _close_on(prices: pd.DataFrame, d: date) -> float | None:
    if prices.empty:
        return None
    mask = prices.index.date == d
    if not mask.any():
        return None
    return float(prices.loc[mask, "Close"].iloc[0])


def compute_call_row(
    call: dict[str, Any],
    stock_prices: pd.DataFrame,
    spy_prices: pd.DataFrame,
) -> dict[str, Any]:
    rd = call["report_date"]
    out: dict[str, Any] = {
        "ticker": call["ticker"],
        "quarter": call["quarter"],
        "year": call["year"],
        "report_date": rd.isoformat(),
        "filename": call["filename"],
    }

    if stock_prices.empty:
        out["status"] = "no_stock_data"
        return out

    anchor = find_trading_day_after(stock_prices, rd)
    if anchor is None:
        out["status"] = "no_anchor"
        return out
    anchor_close = _close_on(stock_prices, anchor)
    if anchor_close is None:
        out["status"] = "no_anchor_close"
        return out

    out["anchor_date"] = anchor.isoformat()
    out["anchor_close"] = anchor_close

    spy_anchor_close = _close_on(spy_prices, anchor) if not spy_prices.empty else None
    out["spy_anchor_close"] = spy_anchor_close

    missing: list[int] = []
    for h in HORIZONS:
        target = shift_n_trading_days(stock_prices, anchor, h)
        if target is None:
            out[f"price_{h}d"] = None
            out[f"ret_{h}d"] = None
            out[f"excess_{h}d"] = None
            missing.append(h)
            continue
        stock_px = _close_on(stock_prices, target)
        out[f"price_{h}d"] = stock_px
        out[f"ret_{h}d"] = stock_px / anchor_close - 1.0 if stock_px is not None else None

        if spy_anchor_close is not None:
            spy_px = _close_on(spy_prices, target)
            if spy_px is not None and out[f"ret_{h}d"] is not None:
                spy_ret = spy_px / spy_anchor_close - 1.0
                out[f"excess_{h}d"] = out[f"ret_{h}d"] - spy_ret
            else:
                out[f"excess_{h}d"] = None
        else:
            out[f"excess_{h}d"] = None

    # Pre-call drift: stock momentum strictly BEFORE report_date.
    # Pre-anchor = last trading day before report_date. Look back 5/21/63 trading days.
    # Look-ahead-safe: never touches report_date or later.
    pre_anchor = find_trading_day_before(stock_prices, rd)
    out["pre_anchor_date"] = pre_anchor.isoformat() if pre_anchor is not None else None
    pre_anchor_close = _close_on(stock_prices, pre_anchor) if pre_anchor is not None else None
    spy_pre_close = _close_on(spy_prices, pre_anchor) if pre_anchor is not None else None

    for h in PRE_CALL_HORIZONS:
        out[f"pre_{h}d_ret"] = None
        out[f"pre_{h}d_excess"] = None
        if pre_anchor is None or pre_anchor_close is None:
            continue
        prior = shift_n_trading_days(stock_prices, pre_anchor, -h)
        if prior is None:
            continue
        prior_close = _close_on(stock_prices, prior)
        if prior_close is None:
            continue
        stock_ret = pre_anchor_close / prior_close - 1.0
        out[f"pre_{h}d_ret"] = stock_ret
        if spy_pre_close is None:
            continue
        prior_spy = shift_n_trading_days(spy_prices, pre_anchor, -h)
        spy_close = _close_on(spy_prices, prior_spy) if prior_spy is not None else None
        if spy_close is None:
            continue
        spy_ret = spy_pre_close / spy_close - 1.0
        out[f"pre_{h}d_excess"] = stock_ret - spy_ret

    out["status"] = "ok" if not missing else f"partial_missing_{'_'.join(str(h) for h in missing)}d"
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase C: download prices and compute forward returns")
    parser.add_argument("--force", action="store_true", help="Re-download cached OHLC")
    args = parser.parse_args(argv)

    PRICES_DIR.mkdir(parents=True, exist_ok=True)
    PRICES_RAW_DIR.mkdir(parents=True, exist_ok=True)

    calls = load_calls()
    if not calls:
        print(f"No extractions in {EXTRACTIONS_DIR}. Run Phase B first.")
        return 1

    tickers = sorted({c["ticker"] for c in calls})
    earliest = min(c["report_date"] for c in calls)
    latest = max(c["report_date"] for c in calls)
    start = earliest - timedelta(days=LOOKBACK_BUFFER_DAYS)
    end = latest + timedelta(days=LOOKAHEAD_BUFFER_DAYS)
    print(f"{len(calls)} calls, {len(tickers)} unique ticker(s), date range {start} -> {end}")

    ticker_prices: dict[str, pd.DataFrame] = {}
    download_failures: list[dict[str, str]] = []
    for t in tqdm(tickers + [BENCHMARK], desc="downloading"):
        try:
            df = download_ohlc(t, start, end, force=args.force)
            ticker_prices[t] = df
            if df.empty:
                download_failures.append({"ticker": t, "error": "empty_dataframe"})
        except Exception as exc:  # noqa: BLE001 — log & continue, never crash batch
            download_failures.append({"ticker": t, "error": str(exc)})
            ticker_prices[t] = pd.DataFrame()

    spy_prices = ticker_prices.get(BENCHMARK, pd.DataFrame())
    rows = [
        compute_call_row(c, ticker_prices.get(c["ticker"], pd.DataFrame()), spy_prices)
        for c in tqdm(calls, desc="computing returns")
    ]

    df_out = pd.DataFrame(rows)
    df_out.to_csv(RETURNS_PATH, index=False)

    by_status: dict[str, int] = {}
    for r in rows:
        s = r.get("status", "?")
        by_status[s] = by_status.get(s, 0) + 1

    summary = {
        "total_calls": len(rows),
        "tickers": tickers,
        "by_status": by_status,
        "download_failures": download_failures,
        "horizons_trading_days": HORIZONS,
        "pre_call_horizons_trading_days": PRE_CALL_HORIZONS,
        "benchmark": BENCHMARK,
        "anchor_rule": "first trading day strictly after report_date (look-ahead-safe)",
        "pre_call_anchor_rule": "last trading day strictly before report_date (look-ahead-safe)",
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Returns written to {RETURNS_PATH} ({len(df_out)} rows)")
    print(f"Status counts: {by_status}")
    if download_failures:
        print(f"Download failures: {download_failures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
