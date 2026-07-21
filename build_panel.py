"""
Stack every weekly snapshot into one long panel: data/panel.csv

Adds derived columns the network wants and a forward-return label, so the
output is directly trainable. Run this locally or in the same workflow.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
SNAP_DIR = ROOT / "data" / "snapshots"
HORIZON_WEEKS = 26         # label horizon


def main() -> None:
    files = sorted(SNAP_DIR.glob("*.csv"))
    if not files:
        raise SystemExit("no snapshots yet")

    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    df["run_date"] = pd.to_datetime(df["run_date"])
    df = df.sort_values(["ticker", "run_date"])

    # derived ratios, recomputed from raw lines so they are consistent across
    # snapshots even when Yahoo changes how it reports its own ratio fields
    df["pb"] = df["price"] / df["book_value_ps"]
    df["pe"] = df["price"] / df["trailing_eps"].where(df["trailing_eps"] > 0)
    df["earnings_yield"] = df["trailing_eps"] / df["price"]
    df["roe_calc"] = df["net_income"] / df["total_equity"]
    df["equity_ps"] = df["total_equity"] / df["shares_out"]

    # forward label: price HORIZON_WEEKS ahead for the same ticker.
    # Relies on one snapshot per week - a skipped run shifts the horizon,
    # so check run_date spacing before trusting this on a long history.
    df["price_fwd"] = df.groupby("ticker")["price"].shift(-HORIZON_WEEKS)
    df["fwd_return"] = df["price_fwd"] / df["price"] - 1

    # winsorise the ratios - one bank with a near-zero equity base will
    # otherwise dominate the loss
    for col in ["pb", "pe", "roe_calc", "earnings_yield"]:
        lo, hi = df[col].quantile([0.01, 0.99])
        df[col] = df[col].clip(lo, hi)

    df = df.replace([np.inf, -np.inf], np.nan)
    out = ROOT / "data" / "panel.csv"
    df.to_csv(out, index=False)
    print(f"{out}: {len(df)} rows, {df['ticker'].nunique()} tickers, "
          f"{df['run_date'].nunique()} dates")


if __name__ == "__main__":
    main()
