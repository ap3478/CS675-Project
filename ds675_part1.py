# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
# ---

# %% [markdown]
# # DS675 Mini Project — Part 1: data, label, features, baselines
#
# **Question:** over the next 5 trading days, will a stock beat the median S&P 500 stock?
#
# This notebook covers Milestone 2 Steps 1–3 (clean, label, features), Step 5 (walk-forward split),
# Step 6 (baselines) and Step 8 (evaluation). LightGBM, the LSTM, peer-group clustering (Step 4)
# and the portfolio simulation (Step 9) build on the `panel` and `folds` produced here.
#
# Design choices that matter for leakage:
# * All prices and returns use **Adj Close**; raw `Close` is only used for dollar volume (Close × Volume),
#   which is split-consistent.
# * The data is pivoted to wide `date × ticker` matrices on one shared trading calendar, so every
#   `shift(k)` means *k trading days* for every stock, and no ticker's window can cross into another's.
# * Every feature at date *t* uses prices up to and including *t* only (checked by a truncation test).
# * All stocks share the same yearly cutoffs, and the last 5 days before each validation/test period
#   are purged so no training label overlaps them.
# * One deliberate exception: the bad-print filter in Step 1 removes a price that jumps >2x and fully
#   reverses the *next* day. This is data cleaning, not a signal, and it touches very few rows (reported).
#
# `tests/test_no_lookahead.py` rebuilds the features on data cut off at several dates and checks they
# are identical to the full-data features, re-derives labels by hand, and checks the fold purges.

# %%
import glob
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)

# %% [markdown]
# ## Configuration

# %%
def find_csv():
    """Env var SP500_CSV wins; otherwise look in Kaggle's input folder, then the current folder."""
    if os.environ.get("SP500_CSV"):
        return os.environ["SP500_CSV"]
    for pattern in ["/kaggle/input/**/*.csv", "./*.csv", "./data/*.csv"]:
        hits = sorted(glob.glob(pattern, recursive=True), key=os.path.getsize, reverse=True)
        if hits:
            return hits[0]  # the main file is by far the largest CSV
    raise FileNotFoundError("Set SP500_CSV or put the dataset CSV in ./ or ./data/")


DATA_PATH = find_csv()
START_DATE = pd.Timestamp("2000-01-01")   # first date that can carry a label / be evaluated
WARMUP_DAYS = 400                          # calendar days of history kept before START_DATE for 252-day features
HORIZON = 5                                # forward return horizon in trading days
MIN_STOCKS_PER_DAY = 50                    # days with fewer valid stocks get no label
EXCLUDE_TICKERS = ["CICC"]                 # removed outright (see Milestone 2, Step 1)
SPIKE_LOG_RET = np.log(2.0)                # a one-day jump of >2x that fully reverses next day = bad print
FLAG_ABS_DAILY_RET = 1.0                   # tickers with any |daily return| > 100% are reported for review
MIN_TRAIN_YEARS = 4                        # first fold trains on 2000–2003, validates 2004, tests 2005
LOGIT_C_GRID = [0.001, 0.01, 0.1, 1.0]
REFIT_ON_TRAIN_VAL = True                  # after picking C on the val year, refit on train+val (purged)
RANDOM_SEED = 675
OUT_DIR = os.environ.get("OUT_DIR", ".")
print("Data file:", DATA_PATH)

# %% [markdown]
# ## Step 1 — Load and clean

# %%
PRICE_COLS = ["Open", "High", "Low", "Close", "Adj Close"]


def load_raw(path):
    cols = ["Date", "Ticker"] + PRICE_COLS + ["Volume"]
    df = pd.read_csv(path, usecols=cols)
    # yfinance dates can be plain or carry a UTC offset; normalise both to a naive calendar date
    df["Date"] = pd.to_datetime(df["Date"], utc=True, format="mixed").dt.tz_localize(None).dt.normalize()
    df["Ticker"] = df["Ticker"].astype(str)
    return df


def clean(raw, start=START_DATE, warmup_days=WARMUP_DAYS, exclude=EXCLUDE_TICKERS, verbose=True):
    rep = {"rows_raw": len(raw), "tickers_raw": raw["Ticker"].nunique()}
    df = raw[~raw["Ticker"].isin(exclude)].copy()
    rep["tickers_excluded"] = sorted(set(exclude) & set(raw["Ticker"].unique()))

    df = df[df["Date"] >= start - pd.Timedelta(days=warmup_days)]
    df = df.sort_values(["Ticker", "Date"])
    n0 = len(df)
    df = df.drop_duplicates(["Ticker", "Date"], keep="last")
    rep["duplicate_rows_dropped"] = n0 - len(df)

    # zero / negative prices are yfinance artefacts -> missing
    for c in PRICE_COLS:
        bad = df[c] <= 0
        rep[f"nonpositive_{c}"] = int(bad.sum())
        df.loc[bad, c] = np.nan
    df.loc[df["Volume"] <= 0, "Volume"] = np.nan

    n0 = len(df)
    df = df.dropna(subset=["Adj Close"])
    rep["rows_without_adj_close_dropped"] = n0 - len(df)

    # isolated bad prints: up >2x then straight back down (or the reverse) -> set that day's prices missing
    lp = np.log(df["Adj Close"])
    r_in = lp - lp.groupby(df["Ticker"]).shift(1)
    r_out = lp.groupby(df["Ticker"]).shift(-1) - lp
    spike = ((r_in > SPIKE_LOG_RET) & (r_out < -SPIKE_LOG_RET)) | ((r_in < -SPIKE_LOG_RET) & (r_out > SPIKE_LOG_RET))
    rep["spike_prints_removed"] = int(spike.sum())
    df = df[~spike]

    # report (not drop) tickers with extreme moves so the group can inspect them
    ret = df.groupby("Ticker")["Adj Close"].pct_change()
    extreme = df.loc[ret.abs() > FLAG_ABS_DAILY_RET, ["Ticker", "Date"]].assign(ret=ret[ret.abs() > FLAG_ABS_DAILY_RET])
    rep["flagged_extreme_moves"] = extreme

    rep["rows_clean"] = len(df)
    rep["tickers_clean"] = df["Ticker"].nunique()
    if verbose:
        for k, v in rep.items():
            if k != "flagged_extreme_moves":
                print(f"{k:32s} {v}")
        print(f"{'flagged_extreme_moves':32s} {len(extreme)} rows in {extreme['Ticker'].nunique()} tickers")
    return df.reset_index(drop=True), rep


def to_wide(df):
    """Pivot to date x ticker matrices on a shared trading calendar."""
    wide = {
        "px": df.pivot(index="Date", columns="Ticker", values="Adj Close"),
        "close": df.pivot(index="Date", columns="Ticker", values="Close"),
        "volume": df.pivot(index="Date", columns="Ticker", values="Volume"),
    }
    # drop calendar days on which only a handful of tickers report (holiday artefacts)
    n = wide["px"].notna().sum(axis=1)
    keep = n >= 0.5 * n.rolling(21, min_periods=1).median()
    for k in wide:
        wide[k] = wide[k].loc[keep].astype("float64")
    return wide


# %% [markdown]
# ## Step 2 — Label
#
# $y_{i,t} = 1[\,r_{i,t\to t+5} > \text{median}_j\, r_{j,t\to t+5}\,]$, using Adj Close and strict ">".

# %%
def make_label(px, horizon=HORIZON, min_stocks=MIN_STOCKS_PER_DAY):
    fwd = px.shift(-horizon) / px - 1.0
    med = fwd.median(axis=1)
    y = fwd.gt(med, axis=0).astype("float64")
    valid = fwd.notna() & (fwd.notna().sum(axis=1) >= min_stocks).to_numpy()[:, None]
    return fwd.where(valid), y.where(valid)


# %% [markdown]
# ## Step 3 — Features (past data only), then cross-sectional ranks

# %%
def _ema(x, span):
    return x.ewm(span=span, adjust=False, min_periods=span).mean()


def make_raw_features(w):
    px, close, vol = w["px"], w["close"], w["volume"]
    lr = np.log(px).diff()
    f = {}
    for k in (1, 5, 21, 63):
        f[f"ret_{k}"] = px / px.shift(k) - 1
    f["mom_12_1"] = px.shift(21) / px.shift(252) - 1            # classic 12-month momentum skipping last month
    f["vol_21"] = lr.rolling(21, min_periods=15).std()
    f["vol_63"] = lr.rolling(63, min_periods=45).std()
    f["vol_ratio"] = f["vol_21"] / f["vol_63"]
    f["max_ret_21"] = lr.rolling(21, min_periods=15).max()      # "lottery" effect
    f["volume_ratio"] = vol / vol.rolling(20, min_periods=15).mean()
    f["dollar_vol_20"] = np.log(close * vol).rolling(20, min_periods=15).mean()  # size / liquidity proxy
    delta = px.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    f["rsi_14"] = 100 - 100 / (1 + gain / loss)
    macd = _ema(px, 12) - _ema(px, 26)
    f["macd"] = macd / px
    f["macd_hist"] = (macd - _ema(macd, 9)) / px
    ma20, sd20 = px.rolling(20, min_periods=20).mean(), px.rolling(20, min_periods=20).std()
    f["bb_pctb"] = (px - ma20) / (2 * sd20)
    f["dist_52w_high"] = px / px.rolling(252, min_periods=126).max() - 1
    alive = px.notna()
    return {k: v.replace([np.inf, -np.inf], np.nan).where(alive) for k, v in f.items()}


def cross_sectional_rank(x):
    """0 = lowest stock that day, 1 = highest; NaN stays NaN."""
    r = x.rank(axis=1, method="average")
    n = x.notna().sum(axis=1)
    return r.sub(1, axis=0).div((n - 1).where(n > 1), axis=0)


FEATURES = [
    "ret_1", "ret_5", "ret_21", "ret_63", "mom_12_1", "vol_21", "vol_63", "vol_ratio",
    "max_ret_21", "volume_ratio", "dollar_vol_20", "rsi_14", "macd", "macd_hist", "bb_pctb", "dist_52w_high",
]


def build_panel(w, start=START_DATE, horizon=HORIZON):
    """Long table: one row per (date, ticker) with a label, the forward return and ranked features."""
    fwd, y = make_label(w["px"], horizon)
    raw = make_raw_features(w)
    ranked = {k: cross_sectional_rank(raw[k]) for k in FEATURES}

    dates, tickers = w["px"].index, w["px"].columns
    mask = (y.notna() & (dates >= start)[:, None]).to_numpy()
    di, ti = np.nonzero(mask)
    panel = pd.DataFrame({
        "date": dates[di], "ticker": tickers[ti], "date_idx": di,
        "y": y.to_numpy()[di, ti].astype("int8"), "fwd_ret": fwd.to_numpy()[di, ti],
    })
    for k in FEATURES:
        panel[k] = ranked[k].to_numpy()[di, ti].astype("float32")
    panel["year"] = panel["date"].dt.year
    return panel


# %% [markdown]
# ## Step 5 — Walk-forward folds (shared yearly cutoffs, 5-day purge)

# %%
def walk_forward_folds(panel, min_train_years=MIN_TRAIN_YEARS, horizon=HORIZON):
    years = sorted(panel["year"].unique())
    first_idx = panel.groupby("year")["date_idx"].min()
    folds = []
    for test_year in years[min_train_years + 1:]:
        val_year = test_year - 1
        if val_year not in first_idx.index:
            continue
        val_start, test_start = first_idx[val_year], first_idx[test_year]
        # a label at t looks at t+horizon, so it must end strictly before the next period starts
        tr = (panel["year"] < val_year) & (panel["date_idx"] + horizon < val_start)
        va = (panel["year"] == val_year) & (panel["date_idx"] + horizon < test_start)
        trva = (panel["year"] < test_year) & (panel["date_idx"] + horizon < test_start)
        te = panel["year"] == test_year
        folds.append({"test_year": test_year, "train": tr.to_numpy(), "val": va.to_numpy(),
                      "train_val": trva.to_numpy(), "test": te.to_numpy()})
    return folds


# %% [markdown]
# ## Step 6 — Baselines, Step 8 — metrics

# %%
def daily_ic(dates, score, fwd):
    """Mean over days of the Spearman correlation between score and forward return."""
    d = pd.DataFrame({"d": dates, "s": score, "f": fwd})
    g = d.groupby("d")
    rs = g["s"].rank() - g["s"].transform("size").add(1) / 2
    rf = g["f"].rank() - g["f"].transform("size").add(1) / 2
    num = (rs * rf).groupby(d["d"]).sum()
    den = np.sqrt((rs ** 2).groupby(d["d"]).sum() * (rf ** 2).groupby(d["d"]).sum())
    return (num / den).dropna()


def evaluate(part, score, horizon=HORIZON):
    d = part[["date", "y", "fwd_ret"]].copy()
    d["s"] = score
    g = d.groupby("date")["s"]
    pred = (d["s"] > g.transform("median")).astype(int)        # top half of each day -> predict 1
    ic = daily_ic(d["date"], d["s"], d["fwd_ret"])
    ic_nonoverlap = ic.iloc[::horizon]
    q = g.rank(pct=True)
    spread = d.loc[q > 0.9].groupby("date")["fwd_ret"].mean() - d.loc[q <= 0.1].groupby("date")["fwd_ret"].mean()
    return {
        "accuracy": (pred == d["y"]).mean(),
        "auc": roc_auc_score(d["y"], d["s"]),
        "ic_mean": ic.mean(),
        "ic_tstat": ic_nonoverlap.mean() / ic_nonoverlap.std() * np.sqrt(len(ic_nonoverlap)),
        "top_minus_bottom_decile_5d_bps": spread.mean() * 1e4,
        "n_rows": len(d),
    }


def X_of(df):
    return (df[FEATURES].fillna(0.5) - 0.5).to_numpy(dtype="float32")


def fit_logit(train, C):
    m = LogisticRegression(C=C, max_iter=300)
    return m.fit(X_of(train), train["y"].to_numpy())


def run_baselines(panel, folds, verbose=True):
    rng = np.random.default_rng(RANDOM_SEED)
    rows, chosen = [], []
    for f in folds:
        t0 = time.time()
        test = panel[f["test"]]
        scores = {
            "random": rng.random(len(test)),
            "momentum_12_1": test["mom_12_1"].fillna(0.5).to_numpy(),
            "momentum_1m": test["ret_21"].fillna(0.5).to_numpy(),
            "reversal_1w": 1 - test["ret_5"].fillna(0.5).to_numpy(),
        }
        train, val = panel[f["train"]], panel[f["val"]]
        best_C = max(LOGIT_C_GRID, key=lambda C: roc_auc_score(
            val["y"], fit_logit(train, C).predict_proba(X_of(val))[:, 1]))
        final = fit_logit(panel[f["train_val"]] if REFIT_ON_TRAIN_VAL else train, best_C)
        scores["logistic"] = final.predict_proba(X_of(test))[:, 1]
        chosen.append({"test_year": f["test_year"], "C": best_C, "coef": dict(zip(FEATURES, final.coef_[0]))})
        for name, s in scores.items():
            rows.append({"test_year": f["test_year"], "model": name, **evaluate(test, s)})
        if verbose:
            print(f"test {f['test_year']}: train {f['train'].sum():>9,d}  val {f['val'].sum():>7,d}  "
                  f"test {f['test'].sum():>7,d}  C={best_C:<6}  {time.time() - t0:5.1f}s")
    return pd.DataFrame(rows), chosen


def summarise(results):
    agg = results.groupby("model").agg(
        accuracy=("accuracy", "mean"), auc=("auc", "mean"), ic_mean=("ic_mean", "mean"),
        years_ic_positive=("ic_mean", lambda s: f"{(s > 0).sum()}/{len(s)}"),
        decile_spread_5d_bps=("top_minus_bottom_decile_5d_bps", "mean"),
    )
    # t-stat of the yearly mean IC across years (years are non-overlapping)
    t = results.groupby("model")["ic_mean"].agg(lambda s: s.mean() / s.std() * np.sqrt(len(s)))
    agg.insert(3, "ic_tstat_across_years", t)
    return agg.sort_values("auc", ascending=False)


# %% [markdown]
# ## Run Steps 1–3

# %%
t0 = time.time()
raw = load_raw(DATA_PATH)
print(f"loaded {len(raw):,} rows in {time.time() - t0:.1f}s")
df, clean_report = clean(raw)
del raw
clean_report["flagged_extreme_moves"].groupby("Ticker").size().sort_values(ascending=False).head(15)

# %%
wide = to_wide(df)
panel = build_panel(wide)
print(f"panel: {len(panel):,} rows, {panel['ticker'].nunique()} tickers, "
      f"{panel['date'].min().date()} to {panel['date'].max().date()}")
per_day = panel.groupby("date").size()
print(f"stocks per day: min {per_day.min()}, median {per_day.median():.0f}, max {per_day.max()}")
print(f"label balance: {panel['y'].mean():.4f}  (≈0.5 by construction, slightly below because of strict '>')")
panel[FEATURES].describe().T[["count", "mean", "min", "max"]]

# %% [markdown]
# Feature coverage and rank IC of each feature on its own (full sample, *descriptive only*;
# model selection uses the walk-forward folds below).

# %%
feat_ic = pd.DataFrame({
    k: {"coverage": panel[k].notna().mean(),
        "ic_mean": daily_ic(panel["date"], panel[k].fillna(0.5), panel["fwd_ret"]).mean()}
    for k in FEATURES}).T
feat_ic.sort_values("ic_mean")

# %% [markdown]
# ## Steps 5–8 — Walk-forward baselines

# %%
folds = walk_forward_folds(panel)
print(f"{len(folds)} folds: test years {folds[0]['test_year']}–{folds[-1]['test_year']}")
results, logit_fits = run_baselines(panel, folds)

# %%
summary = summarise(results)
summary

# %%
results.pivot(index="test_year", columns="model", values="auc").round(4)

# %%
coef = pd.DataFrame({c["test_year"]: c["coef"] for c in logit_fits}).T
print("Logistic coefficients (feature ranks centred at 0), last 5 folds:")
coef.tail(5).T.round(3)

# %%
try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    for m, g in results.groupby("model"):
        ax[0].plot(g["test_year"], g["auc"], marker="o", ms=3, label=m)
        ax[1].plot(g["test_year"], g["ic_mean"], marker="o", ms=3, label=m)
    ax[0].axhline(0.5, color="grey", lw=0.8); ax[1].axhline(0, color="grey", lw=0.8)
    ax[0].set_title("AUC by test year"); ax[1].set_title("Mean daily rank IC by test year")
    ax[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "baselines_by_year.png"), dpi=120)
    plt.show()
except ImportError:
    pass

# %%
os.makedirs(OUT_DIR, exist_ok=True)
results.to_csv(os.path.join(OUT_DIR, "baseline_results_by_year.csv"), index=False)
summary.to_csv(os.path.join(OUT_DIR, "baseline_summary.csv"))
panel.to_parquet(os.path.join(OUT_DIR, "panel.parquet")) if os.environ.get("SAVE_PANEL") else None
print("saved results to", os.path.abspath(OUT_DIR))
