"""Synthetic S&P-500-style CSV with the real file's schema and quirks, for testing the pipeline."""
import sys
import numpy as np
import pandas as pd

out, n_tickers, start, end = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
rng = np.random.default_rng(0)
dates = pd.bdate_range(start, end)
T = len(dates)
mkt = rng.normal(0.0003, 0.011, T)
rows = []
for i in range(n_tickers):
    tk = f"T{i:03d}"
    first = 0 if i < n_tickers // 2 else rng.integers(0, T // 2)       # late IPOs -> unbalanced panel
    n = T - first
    beta = rng.uniform(0.6, 1.5)
    sig = rng.uniform(0.01, 0.03)
    eps = rng.normal(0, sig, n)
    # planted signal: weak reversal of the last 5 days' idiosyncratic return
    past5 = pd.Series(eps).rolling(5).sum().shift(1).fillna(0).to_numpy()
    r = beta * mkt[first:] + eps - 0.04 * past5
    adj = rng.uniform(5, 200) * np.exp(np.cumsum(r))
    split_factor = np.ones(n)                                         # raw Close jumps on split days
    for s in rng.choice(np.arange(50, n), size=rng.integers(0, 3), replace=False) if n > 60 else []:
        split_factor[:s] *= rng.choice([2, 3, 10])
    close = adj * split_factor * rng.uniform(1.0, 1.05)              # Close != Adj Close (dividends)
    vol = rng.lognormal(14, 0.6, n).round()
    d = pd.DataFrame({
        "Date": dates[first:].strftime("%Y-%m-%d"), "Ticker": tk,
        "Open": close * (1 + rng.normal(0, 0.003, n)), "High": close * 1.01, "Low": close * 0.99,
        "Close": close, "Adj Close": adj, "Volume": vol.astype("int64"),
        "Stock Splits": 0.0, "Dividends": 0.0,
    })
    d.loc[rng.random(n) < 0.002, "Open"] = 0.0                        # zero-open artefacts
    d.loc[rng.random(n) < 0.0005, ["Low"]] = 0.0
    d = d.drop(index=d.sample(frac=0.002, random_state=i).index)     # random missing days
    rows.append(d)
df = pd.concat(rows)
df = pd.concat([df, df.sample(500, random_state=1)])                 # duplicate rows
bad = df.sample(3, random_state=2).index                              # single-day bad prints
df.loc[bad, "Adj Close"] *= 5
junk = pd.DataFrame({"Date": dates[-300:].strftime("%Y-%m-%d"), "Ticker": "CICC", "Open": 1.0, "High": 1.0,
                     "Low": 1.0, "Close": 1.0, "Adj Close": 1.0, "Volume": 0, "Stock Splits": 0.0, "Dividends": 0.0})
df = pd.concat([df, junk]).sample(frac=1, random_state=3)             # unsorted, like a raw dump
df.to_csv(out, index=False)
print(f"wrote {len(df):,} rows, {df['Ticker'].nunique()} tickers to {out}")
