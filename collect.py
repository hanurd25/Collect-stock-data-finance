"""
Weekly point-in-time fundamentals snapshot for sector baskets.

Every run writes ONE immutable file: data/snapshots/<run_date>.csv
Nothing is ever overwritten. That is the whole point - Yahoo only serves the
*current* state of each field, so the only way to get a point-in-time history
is to stamp it yourself every week and never touch it again.

Usage:
    python collect.py                 # full run, writes a snapshot
    python collect.py --check         # validate tickers only, no snapshot
    python collect.py --baskets banking_nordic insurance_nordic
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import random
import sys
import time
from pathlib import Path

import pandas as pd
import yaml
import yfinance as yf

ROOT = Path(__file__).parent
SNAP_DIR = ROOT / "data" / "snapshots"

MIN_SLEEP, MAX_SLEEP = 1.5, 3.5     # polite pause between tickers
MAX_RETRIES = 4

log = logging.getLogger("collect")


# --------------------------------------------------------------------------
# field extraction
# --------------------------------------------------------------------------
def _first(d: dict, *keys):
    """First non-missing key. Yahoo renames fields without warning."""
    for k in keys:
        v = d.get(k)
        if v is not None and not (isinstance(v, float) and pd.isna(v)):
            return v
    return None


def _stmt_value(df, *row_names):
    """Most recent value of a row in a yfinance statement DataFrame."""
    if df is None or df.empty:
        return None
    for name in row_names:
        if name in df.index:
            series = df.loc[name].dropna()
            if not series.empty:
                return float(series.iloc[0])
    return None


def extract(ticker: str, basket: str) -> dict:
    tk = yf.Ticker(ticker)
    info = tk.info or {}

    try:
        bs = tk.balance_sheet
    except Exception:
        bs = None
    try:
        inc = tk.income_stmt
    except Exception:
        inc = None

    equity = _stmt_value(bs, "Common Stock Equity", "Stockholders Equity",
                         "Total Stockholder Equity")
    net_income = _stmt_value(inc, "Net Income", "Net Income Common Stockholders")
    revenue = _stmt_value(inc, "Total Revenue", "Operating Revenue")

    shares = _first(info, "sharesOutstanding", "impliedSharesOutstanding")
    price = _first(info, "currentPrice", "regularMarketPrice", "previousClose")

    book_ps = _first(info, "bookValue")
    if book_ps is None and equity and shares:
        book_ps = equity / shares

    roe = _first(info, "returnOnEquity")
    if roe is None and net_income and equity:
        roe = net_income / equity

    return {
        "ticker": ticker,
        "basket": basket,
        "sector_label": basket.rsplit("_", 1)[0],      # banking / insurance
        "region": basket.rsplit("_", 1)[-1],
        "name": _first(info, "shortName", "longName"),
        "currency": _first(info, "currency"),
        # --- market state ---
        "price": price,
        "market_cap": _first(info, "marketCap"),
        "shares_out": shares,
        "beta": _first(info, "beta"),
        # --- per share ---
        "trailing_eps": _first(info, "trailingEps"),
        "forward_eps": _first(info, "forwardEps"),
        "book_value_ps": book_ps,
        # --- ratios Yahoo computes ---
        "price_to_book": _first(info, "priceToBook"),
        "trailing_pe": _first(info, "trailingPE"),
        "forward_pe": _first(info, "forwardPE"),
        "roe": roe,
        "roa": _first(info, "returnOnAssets"),
        "profit_margin": _first(info, "profitMargins"),
        "operating_margin": _first(info, "operatingMargins"),
        "revenue_growth": _first(info, "revenueGrowth"),
        "earnings_growth": _first(info, "earningsGrowth",
                                  "earningsQuarterlyGrowth"),
        "dividend_yield": _first(info, "dividendYield"),
        "payout_ratio": _first(info, "payoutRatio"),
        "debt_to_equity": _first(info, "debtToEquity"),
        # --- raw statement lines, so ratios can be recomputed later ---
        "total_equity": equity,
        "net_income": net_income,
        "revenue": revenue,
        # --- provenance ---
        "last_fiscal_year_end": str(_first(info, "lastFiscalYearEnd") or ""),
        "most_recent_quarter": str(_first(info, "mostRecentQuarter") or ""),
    }


def extract_with_retry(ticker: str, basket: str) -> dict | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            row = extract(ticker, basket)
            if row.get("price") is None:
                raise ValueError("no price field - likely a bad ticker or a block")
            return row
        except Exception as exc:
            wait = min(60, 2 ** attempt) + random.uniform(0, 3)
            log.warning("%s attempt %d/%d failed (%s); sleeping %.1fs",
                        ticker, attempt, MAX_RETRIES, exc, wait)
            time.sleep(wait)
    log.error("%s: giving up", ticker)
    return None


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="validate tickers, do not write a snapshot")
    ap.add_argument("--baskets", nargs="*", default=None)
    ap.add_argument("--universe", default="universe.yaml")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    universe = yaml.safe_load((ROOT / args.universe).read_text())
    if args.baskets:
        universe = {k: v for k, v in universe.items() if k in args.baskets}

    run_date = dt.date.today().isoformat()
    rows, failed = [], []

    for basket, tickers in universe.items():
        log.info("basket %s (%d tickers)", basket, len(tickers))
        for ticker in tickers:
            row = extract_with_retry(ticker, basket)
            if row is None:
                failed.append(ticker)
            else:
                rows.append(row)
            time.sleep(random.uniform(MIN_SLEEP, MAX_SLEEP))

    if not rows:
        log.error("no rows collected at all - aborting without writing")
        return 1

    df = pd.DataFrame(rows)
    df.insert(0, "run_date", run_date)

    n_total = len(rows) + len(failed)
    coverage = len(rows) / n_total
    log.info("collected %d/%d tickers (%.0f%%)", len(rows), n_total, 100 * coverage)
    if failed:
        log.warning("failed: %s", ", ".join(failed))

    if args.check:
        missing = df.isna().mean().sort_values(ascending=False)
        print("\nmissing rate per field:")
        print(missing[missing > 0].to_string())
        return 0 if coverage > 0.8 else 1

    # A snapshot with big holes is worse than no snapshot - it silently
    # poisons the panel later.
    if coverage < 0.6:
        log.error("coverage %.0f%% below threshold - refusing to write", 100 * coverage)
        return 1

    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    out = SNAP_DIR / f"{run_date}.csv"
    df.to_csv(out, index=False)
    log.info("wrote %s (%d rows, %d cols)", out, len(df), df.shape[1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
