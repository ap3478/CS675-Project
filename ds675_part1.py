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
# # DS675 Mini Project — Part 1: data, label, features, baselines, backtest, PCA, LightGBM
#
# **Question:** over the next 5 trading days, will a stock beat the median S&P 500 stock?
#
# This notebook covers Milestone 2 Steps 1–3 (clean, label, features), Step 5 (walk-forward split),
# Step 6 (baselines) and Step 8 (evaluation), then saves the out-of-sample predictions for every
# stock and day and runs a first weekly long-short backtest with trading costs (Step 9).
# Then a PCA analysis of the features and of the stocks' daily returns, and finally LightGBM
# (Step 7) on the same folds, compared with logistic regression and combined into an ensemble.
# The LSTM and peer-group clustering (Step 4) build on the `panel` and `folds` produced here.
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
# %% [markdown]
# ## Predicting the past: out-of-sample predictions and a backtest
#
# Every prediction below comes from a model trained only on years *before* the year it predicts
# (the same walk-forward folds as above), so this is an honest replay of history.
# Never train on all years and then predict those same years: that model has already seen the answers.

# %%
def oos_predictions(panel, folds, logit_fits):
    """Refit each fold's logistic model (same C as chosen above) and keep its test-year predictions."""
    out = []
    for f, fit in zip(folds, logit_fits):
        train = panel[f["train_val"]] if REFIT_ON_TRAIN_VAL else panel[f["train"]]
        model = fit_logit(train, fit["C"])
        test = panel.loc[f["test"], ["date", "ticker", "y", "fwd_ret"]].copy()
        test["prob_beat_median"] = model.predict_proba(X_of(panel[f["test"]]))[:, 1]
        out.append(test)
    preds = pd.concat(out, ignore_index=True)
    g = preds.groupby("date")["prob_beat_median"]
    preds["rank_pct"] = g.rank(pct=True)                       # 1 = model's favourite stock that day
    preds["predicted_y"] = (preds["prob_beat_median"] > g.transform("median")).astype(int)
    preds["correct"] = (preds["predicted_y"] == preds["y"]).astype(int)
    return preds


t0 = time.time()
preds = oos_predictions(panel, folds, logit_fits)
print(f"{len(preds):,} out-of-sample predictions, {preds['date'].min().date()} to {preds['date'].max().date()} "
      f"({time.time() - t0:.0f}s); overall hit rate {preds['correct'].mean():.4f}")
preds.to_csv(os.path.join(OUT_DIR, "oos_predictions.csv.gz"), index=False)
preds.head()

# %% [markdown]
# ### Look up one stock
# `y` = what actually happened (1 = beat the median over the next 5 days),
# `predicted_y` = the model's call, `fwd_ret` = the stock's actual 5-day return.

# %%
TICKER, FROM, TO = "AAPL", "2020-02-15", "2022-04-15"   # change the ticker and dates here
if TICKER not in set(preds["ticker"]):
    print(f"{TICKER} not in the data; using {preds['ticker'].iloc[0]} instead")
    TICKER = preds["ticker"].iloc[0]
preds[(preds["ticker"] == TICKER) & preds["date"].between(FROM, TO)][
    ["date", "ticker", "prob_beat_median", "rank_pct", "predicted_y", "y", "fwd_ret", "correct"]]

# %%
# Which stocks did the model call best and worst on one day?
DAY = preds["date"].iloc[len(preds) // 2]
day = preds[preds["date"] == DAY].sort_values("prob_beat_median", ascending=False)
print(f"{DAY.date()}: model's top 5 and bottom 5, with what actually happened over the next 5 days")
pd.concat([day.head(5), day.tail(5)])[["ticker", "prob_beat_median", "fwd_ret", "y"]]

# %%
by_ticker = preds.groupby("ticker").agg(days=("correct", "size"), hit_rate=("correct", "mean"))
print("Hit rate per stock (all should hover near 0.5; the edge is small and spread across many stocks):")
by_ticker.describe().T

# %% [markdown]
# ### Check the labels for the looked-up stock against the median stock
# For each day: the stock's actual 5-day return, the median stock's 5-day return, and whether the stock
# beat it. `labels match recomputation` must be True (it re-derives `y` from the returns).

# %%
a = preds[(preds["ticker"] == TICKER) & preds["date"].between(FROM, TO)].copy()
med = preds.groupby("date")["fwd_ret"].median()
a["median_stock_5d"] = a["date"].map(med)
a["beat_median_check"] = (a["fwd_ret"] > a["median_stock_5d"]).astype(int)
print(f"{TICKER} {FROM} to {TO}: {len(a)} days, model right {a['correct'].mean():.1%}; "
      f"labels match recomputation: {(a['beat_median_check'] == a['y']).all()}")
a[["date", "rank_pct", "predicted_y", "fwd_ret", "median_stock_5d", "y", "correct"]]

# %% [markdown]
# ### Is the model more accurate when it is more confident about this stock?
# Days are grouped by where the model ranked the stock that day. For one stock the groups are small,
# so read this with care (each label covers 5 days, so 100 days is only about 20 independent calls).

# %%
a["confidence"] = pd.cut(a["rank_pct"], [0, 0.2, 0.4, 0.6, 0.8, 1.0],
                         labels=["strong lag", "lag", "neutral", "beat", "strong beat"])
print(a.groupby("confidence", observed=True).agg(days=("correct", "size"),
                                                  hit_rate=("correct", "mean")).round(3))
print(f"\n{TICKER} actually beat the median on {a['y'].mean():.1%} of these days")
print(f"{TICKER} hit rate 2005–2026: {by_ticker.loc[TICKER, 'hit_rate']:.3f} "
      f"(average stock {by_ticker['hit_rate'].mean():.3f})")

# %% [markdown]
# ### Which stocks does the model get most and least right?
# With 231 stocks, the extremes are expected to be spread out by chance alone: about ±1.5 points per stock
# (each stock has roughly 1,100 independent 5-day periods), and with 231 stocks one landing ~3 standard
# deviations from the average happens by luck roughly a quarter of the time.

# %%
ranked = by_ticker.sort_values("hit_rate")
print(f"hit-rate spread across stocks: {by_ticker['hit_rate'].std():.4f} "
      f"(chance alone ≈ {np.sqrt(0.25 / (by_ticker['days'].median() / HORIZON)):.4f})")
pd.concat({"worst 10": ranked.head(10).reset_index(), "best 10": ranked.tail(10)[::-1].reset_index()},
          axis=1).round(3)

# %% [markdown]
# ### Does accuracy depend on company size?
# Stocks are split into five equal groups by their dollar-volume rank that day (a size/liquidity proxy).
# `model_said_beat` vs `actually_beat` shows the model's size tilt: it favours small stocks much more
# than the data justifies. Accuracy is still flat across groups, but the backtest's long-short
# portfolio inherits this tilt (long small, short large).

# %%
x = preds.merge(panel[["date", "ticker", "dollar_vol_20"]], on=["date", "ticker"])
x["size"] = pd.qcut(x["dollar_vol_20"], 5, labels=["smallest", "small", "mid", "large", "largest"])
x.groupby("size", observed=True).agg(hit_rate=("correct", "mean"),
                                     actually_beat=("y", "mean"),
                                     model_said_beat=("predicted_y", "mean")).round(3)

# %% [markdown]
# ### Backtest (Milestone Step 9, first version)
# Every 5th trading day: buy the model's top 10% (equal weight), short its bottom 10%, hold 5 days.
# Trading cost: 0.10% of every dollar traded. Also shown: long-only top 10%, and the equal-weight
# average of all stocks as a benchmark. Returns use Adj Close, so dividends are included.

# %%
COST_PER_TRADE = 0.001
DECILE = 0.10


def backtest(preds, decile=DECILE, cost=COST_PER_TRADE, step=HORIZON):
    rebal_dates = np.sort(preds["date"].unique())[::step]      # non-overlapping 5-day holding periods
    prev_long, prev_short, rows = pd.Series(dtype=float), pd.Series(dtype=float), []
    for d in rebal_dates:
        day = preds[preds["date"] == d].dropna(subset=["fwd_ret"])
        q = day["prob_beat_median"].rank(pct=True)
        long_, short_ = day[q > 1 - decile], day[q <= decile]
        w_long = pd.Series(1 / len(long_), index=long_["ticker"].to_numpy())
        w_short = pd.Series(1 / len(short_), index=short_["ticker"].to_numpy())
        # dollars traded = how much the weights changed since last period (sells + buys)
        t_long = w_long.sub(prev_long, fill_value=0).abs().sum()
        t_short = w_short.sub(prev_short, fill_value=0).abs().sum()
        r_long, r_short = long_["fwd_ret"].mean(), short_["fwd_ret"].mean()
        rows.append({
            "date": pd.Timestamp(d),
            "long_short_gross": r_long - r_short,
            "long_short_net": r_long - r_short - cost * (t_long + t_short),
            "long_only_net": r_long - cost * t_long,
            "all_stocks": day["fwd_ret"].mean(),
            "turnover": (t_long + t_short) / 4,                 # share of the book replaced, 0..1
        })
        prev_long, prev_short = w_long, w_short
    return pd.DataFrame(rows).set_index("date")


def perf(r, periods_per_year=252 / HORIZON):
    ann_ret = (1 + r).prod() ** (periods_per_year / len(r)) - 1
    ann_vol = r.std() * np.sqrt(periods_per_year)
    wealth = (1 + r).cumprod()
    return pd.Series({"annual_return": ann_ret, "annual_vol": ann_vol,
                      "sharpe": r.mean() / r.std() * np.sqrt(periods_per_year),
                      "max_drawdown": (wealth / wealth.cummax() - 1).min(),
                      "weeks_positive": (r > 0).mean()})


bt = backtest(preds)
strategies = ["long_short_gross", "long_short_net", "long_only_net", "all_stocks"]
print(f"{len(bt)} weekly periods; average turnover {bt['turnover'].mean():.0%} of the book per week")
pd.DataFrame({s: perf(bt[s]) for s in strategies}).T.round(3)

# %%
yearly = bt[strategies].groupby(bt.index.year).apply(lambda x: (1 + x).prod() - 1)
yearly.round(3)

# %%
try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for s in strategies:
        ax.plot((1 + bt[s]).cumprod(), label=s)
    ax.set_yscale("log"); ax.set_title("Growth of $1, out-of-sample 2005–2026 (log scale)")
    ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "backtest_growth.png"), dpi=120)
    plt.show()
except ImportError:
    pass

bt.to_csv(os.path.join(OUT_DIR, "backtest_weekly.csv"))
# %% [markdown]
# ## PCA — Principal Component Analysis
#
# PCA rewrites a set of correlated variables as new, uncorrelated variables ("principal components",
# PC1, PC2, ...), ordered by how much of the total variation each one explains. It answers
# "how many independent things are really going on here, and what are they?"
#
# We use it twice:
# * **A. On the 16 features** — they overlap heavily (RSI, Bollinger %B, MACD and recent returns all
#   measure "has the stock been rising?"). PCA shows how many independent signals the model really has,
#   explains why individual logistic coefficients are hard to interpret, and tests whether a model on a
#   few components predicts as well as one on all 16.
# * **B. On daily stock returns** — PC1 is the whole market moving together; the next components are
#   sector-like groups (e.g. energy vs. tech). This is the hidden structure behind the sector bets we saw
#   on 2016-04-26, and the basis for the Step 4 peer groups, since the dataset has no sector column.

# %%
from sklearn.decomposition import PCA

# %% [markdown]
# ### A. PCA on the 16 features (descriptive, full sample)

# %%
Xf = panel[FEATURES].fillna(0.5).to_numpy("float64")
Xf = (Xf - Xf.mean(0)) / Xf.std(0)
pca_f = PCA().fit(Xf)
pc_names = [f"PC{i + 1}" for i in range(len(FEATURES))]
feat_var = pd.DataFrame({"explained": pca_f.explained_variance_ratio_,
                         "cumulative": pca_f.explained_variance_ratio_.cumsum()}, index=pc_names)
for share in (0.8, 0.9):
    print(f"components needed for {share:.0%} of the variation in 16 features: "
          f"{int((feat_var['cumulative'] < share).sum()) + 1}")
feat_var.round(3)

# %%
print("Loadings: how much each feature contributes to each component (sign only matters within a column)")
feat_load = pd.DataFrame(pca_f.components_[:5].T, index=FEATURES, columns=pc_names[:5])
feat_load.round(2)

# %%
try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    corr = panel[FEATURES].corr()
    im = ax[0].imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax[0].set_xticks(range(len(FEATURES)), FEATURES, rotation=90, fontsize=8)
    ax[0].set_yticks(range(len(FEATURES)), FEATURES, fontsize=8)
    ax[0].set_title("Feature correlations")
    fig.colorbar(im, ax=ax[0], fraction=0.046)
    ax[1].bar(range(1, len(FEATURES) + 1), feat_var["explained"], label="each component")
    ax[1].plot(range(1, len(FEATURES) + 1), feat_var["cumulative"], "o-", color="black", label="cumulative")
    ax[1].axhline(0.9, color="grey", lw=0.8, ls="--")
    ax[1].set_xlabel("component"); ax[1].set_ylabel("share of variance"); ax[1].set_title("Scree plot")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "pca_features.png"), dpi=120)
    plt.show()
except ImportError:
    pass

# %% [markdown]
# ### A2. Does a model on k components predict as well as one on all 16 features?
# PCA is refit **inside each fold on training data only** (fitting it on all years would leak the
# future into the components). To keep this quick, training uses every 5th day (labels don't overlap)
# and a fixed C; `k = 16` is the same logistic model on all features, for comparison.

# %%
PCA_K_LIST = [1, 2, 3, 5, 8, 16]
PCA_C = 1.0


def pca_logit_scores(train, test, k, C=PCA_C):
    Xtr, Xte = X_of(train).astype("float64"), X_of(test).astype("float64")
    mu, sd = Xtr.mean(0), Xtr.std(0)
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    if k < Xtr.shape[1]:
        p = PCA(n_components=k).fit(Xtr)
        Xtr, Xte = p.transform(Xtr), p.transform(Xte)
    m = LogisticRegression(C=C, max_iter=500).fit(Xtr, train["y"].to_numpy())
    return m.predict_proba(Xte)[:, 1]


t0 = time.time()
pca_rows = []
for f in folds:
    train = panel[f["train_val"]]
    train = train[train["date_idx"] % HORIZON == 0]
    test = panel[f["test"]]
    for k in PCA_K_LIST:
        pca_rows.append({"test_year": f["test_year"], "k": k,
                         **evaluate(test, pca_logit_scores(train, test, k))})
pca_results = pd.DataFrame(pca_rows)
print(f"done in {time.time() - t0:.0f}s")
pca_results.groupby("k").agg(
    auc=("auc", "mean"), ic_mean=("ic_mean", "mean"),
    years_ic_positive=("ic_mean", lambda s: f"{(s > 0).sum()}/{len(s)}"),
    decile_spread_5d_bps=("top_minus_bottom_decile_5d_bps", "mean")).round(4)

# %% [markdown]
# ### B. PCA on daily stock returns: the market's hidden structure

# %%
log_ret = np.log(wide["px"]).diff()
RET_WINDOW_START = str(log_ret.index.max() - pd.DateOffset(years=2))   # last two years of data
R = log_ret.loc[RET_WINDOW_START:]
R = R.loc[:, R.notna().mean() > 0.98].fillna(0.0)
R = R.loc[:, R.std() > 0]
Z = (R - R.mean()) / R.std()                                            # correlation-based PCA
pca_r = PCA().fit(Z.to_numpy())
ret_load = pd.DataFrame(pca_r.components_[:4].T, index=R.columns, columns=["PC1", "PC2", "PC3", "PC4"])
ret_load *= np.sign(ret_load.mean())                                    # orient each PC
ev_r = pca_r.explained_variance_ratio_
print(f"{R.shape[1]} stocks, {R.shape[0]} days from {R.index.min().date()} to {R.index.max().date()}")
print(f"PC1 (the market) explains {ev_r[0]:.1%} of daily return variation; "
      f"PCs 1-5: {ev_r[:5].sum():.1%}; PCs 1-10: {ev_r[:10].sum():.1%}")
print(f"stocks with a positive PC1 loading: {(ret_load['PC1'] > 0).mean():.0%} "
      "(all moving with the market)")

# %%
for pc in ["PC2", "PC3", "PC4"]:
    s = ret_load[pc].sort_values()
    print(f"{pc}  one end: {', '.join(s.index[:8])}\n     other end: {', '.join(s.index[-8:][::-1])}\n")

# %% [markdown]
# Nearest "peers" of a stock: stocks whose returns load most similarly on PCs 2–11
# (PC1 is left out because every stock moves with the market). This is how Step 4 can learn
# sector-like groups without a sector column.

# %%
PEER_OF = "AAPL" if "AAPL" in R.columns else R.columns[0]
F = pca_r.components_[1:11].T * np.sqrt(pca_r.explained_variance_[1:11])
F = F / np.linalg.norm(F, axis=1, keepdims=True)
peer_sim = pd.DataFrame(F @ F.T, index=R.columns, columns=R.columns)
print(f"Most similar stocks to {PEER_OF}:")
peer_sim[PEER_OF].drop(PEER_OF).nlargest(10).round(3)

# %% [markdown]
# How strongly does the whole market move together, year by year? (PC1 share of each year's returns.)
# It jumps in crises, when "everything falls together"; that is when stock-picking signals have the
# least room to work and the relative label matters most.

# %%
pc1_rows = []
for yr, g in log_ret.groupby(log_ret.index.year):
    g = g.loc[:, g.notna().mean() > 0.95].fillna(0.0)
    g = g.loc[:, g.std() > 0]
    if g.shape[1] < 50 or g.shape[0] < 100:
        continue
    eig = np.linalg.eigvalsh(np.corrcoef(g.to_numpy().T))
    pc1_rows.append({"year": yr, "pc1_share": eig[-1] / eig.sum(), "stocks": g.shape[1]})
pc1_by_year = pd.DataFrame(pc1_rows).set_index("year")
pc1_by_year.round(3).T

# %%
try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].bar(pc1_by_year.index, pc1_by_year["pc1_share"])
    ax[0].set_title("Share of daily return variation explained by PC1 (market), by year")
    x, y = ret_load["PC2"], ret_load["PC3"]
    ax[1].scatter(x, y, s=10, alpha=0.6)
    extreme = (x.abs() + y.abs()).nlargest(25).index
    for t in extreme:
        ax[1].annotate(t, (x[t], y[t]), fontsize=7)
    ax[1].axhline(0, color="grey", lw=0.5); ax[1].axvline(0, color="grey", lw=0.5)
    ax[1].set_xlabel("PC2 loading"); ax[1].set_ylabel("PC3 loading")
    ax[1].set_title(f"Stocks on PC2 vs PC3 (last two years)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "pca_returns.png"), dpi=120)
    plt.show()
except ImportError:
    pass

pca_results.to_csv(os.path.join(OUT_DIR, "pca_logit_results_by_year.csv"), index=False)
# %% [markdown]
# ## Step 7a — LightGBM (gradient-boosted trees)
#
# LightGBM builds hundreds of small decision trees, each one correcting the errors of the previous ones.
# Unlike logistic regression it can learn **non-linear effects** ("reversal only matters for volatile
# stocks") and **interactions** between features, and it handles missing values natively.
#
# Same walk-forward folds as the baselines, so the comparison is fair:
# * train on years ≤ Y, use year Y+1 only to decide **when to stop adding trees** (early stopping on AUC),
# * then refit on years ≤ Y+1 with that number of trees and predict year Y+2 (same as `REFIT_ON_TRAIN_VAL`).
#
# Stock returns are mostly noise, so the settings are deliberately conservative: small learning rate,
# large minimum leaf size, row/feature subsampling and L2 regularisation. This keeps the trees from
# memorising noise.

# %%
import lightgbm as lgb

LGB_PARAMS = dict(
    objective="binary", metric="auc",
    learning_rate=0.05, num_leaves=31, max_depth=-1,
    min_child_samples=2000,          # every leaf must cover >= 2,000 stock-days
    feature_fraction=0.8, bagging_fraction=0.7, bagging_freq=1,
    lambda_l2=10.0, verbose=-1, seed=RANDOM_SEED, deterministic=True, force_row_wise=True,
)
LGB_MAX_ROUNDS = 1000
LGB_EARLY_STOP = 50


def lgb_data(df):
    return lgb.Dataset(df[FEATURES].to_numpy("float32"), label=df["y"].to_numpy(), free_raw_data=True)


def run_lightgbm(panel, folds, verbose=True):
    out_preds, out_rows, importances = [], [], []
    for f in folds:
        t0 = time.time()
        train, val, test = panel[f["train"]], panel[f["val"]], panel[f["test"]]
        # 1) pick the number of trees on the validation year
        booster = lgb.train(LGB_PARAMS, lgb_data(train), num_boost_round=LGB_MAX_ROUNDS,
                            valid_sets=[lgb_data(val)],
                            callbacks=[lgb.early_stopping(LGB_EARLY_STOP, verbose=False)])
        n_trees = max(booster.best_iteration, 10)
        val_auc = booster.best_score["valid_0"]["auc"]
        # 2) refit on train + validation with that many trees, predict the test year
        if REFIT_ON_TRAIN_VAL:
            booster = lgb.train(LGB_PARAMS, lgb_data(panel[f["train_val"]]), num_boost_round=n_trees)
        score = booster.predict(test[FEATURES].to_numpy("float32"), num_iteration=n_trees)
        p = test[["date", "ticker", "y", "fwd_ret"]].copy()
        p["prob_beat_median"] = score
        out_preds.append(p)
        out_rows.append({"test_year": f["test_year"], "model": "lightgbm", **evaluate(test, score)})
        importances.append(pd.Series(booster.feature_importance("gain"), index=FEATURES, name=f["test_year"]))
        if verbose:
            print(f"test {f['test_year']}: {n_trees:4d} trees, val AUC {val_auc:.4f}, "
                  f"test AUC {out_rows[-1]['auc']:.4f}  ({time.time() - t0:5.1f}s)")
    preds_lgb = pd.concat(out_preds, ignore_index=True)
    g = preds_lgb.groupby("date")["prob_beat_median"]
    preds_lgb["rank_pct"] = g.rank(pct=True)
    preds_lgb["predicted_y"] = (preds_lgb["prob_beat_median"] > g.transform("median")).astype(int)
    preds_lgb["correct"] = (preds_lgb["predicted_y"] == preds_lgb["y"]).astype(int)
    return preds_lgb, pd.DataFrame(out_rows), pd.concat(importances, axis=1)


t0 = time.time()
preds_lgb, lgb_results, lgb_importance = run_lightgbm(panel, folds)
print(f"LightGBM walk-forward done in {(time.time() - t0) / 60:.1f} min")

# %% [markdown]
# ### Ensemble: average of the two models' daily ranks
# Logistic regression and LightGBM make partly different mistakes, so averaging their rankings
# is a cheap way to get a steadier signal.

# %%
ens = preds[["date", "ticker", "y", "fwd_ret", "rank_pct"]].rename(columns={"rank_pct": "r_logit"}).merge(
    preds_lgb[["date", "ticker", "rank_pct"]].rename(columns={"rank_pct": "r_lgb"}), on=["date", "ticker"])
ens["prob_beat_median"] = 0.5 * ens["r_logit"] + 0.5 * ens["r_lgb"]
ens_rows = []
for f in folds:
    part = ens[ens["date"].dt.year == f["test_year"]]
    ens_rows.append({"test_year": f["test_year"], "model": "ensemble_logit_lgbm",
                     **evaluate(part, part["prob_beat_median"].to_numpy())})
g = ens.groupby("date")["prob_beat_median"]
ens["rank_pct"] = g.rank(pct=True)
ens["predicted_y"] = (ens["prob_beat_median"] > g.transform("median")).astype(int)
ens["correct"] = (ens["predicted_y"] == ens["y"]).astype(int)

# %% [markdown]
# ### Results: all models, 22 out-of-sample test years

# %%
all_results = pd.concat([results, lgb_results, pd.DataFrame(ens_rows)], ignore_index=True)
summary_all = summarise(all_results)
summary_all

# %%
by_year = all_results.pivot(index="test_year", columns="model", values="auc")
by_year["lgbm_minus_logit"] = by_year["lightgbm"] - by_year["logistic"]
print(f"LightGBM beats logistic AUC in {(by_year['lgbm_minus_logit'] > 0).sum()}/{len(by_year)} years")
by_year[["logistic", "lightgbm", "ensemble_logit_lgbm", "lgbm_minus_logit"]].round(4)

# %% [markdown]
# ### Which features does LightGBM rely on?
# "Gain" = total improvement in the training objective from splits on that feature, as a share of all
# gain, averaged over the 22 yearly models. High gain means the feature is *used*, not that it causes
# returns, and overlapping features (see the PCA section) share credit between them.

# %%
imp = lgb_importance.div(lgb_importance.sum(axis=0), axis=1)
imp_summary = pd.DataFrame({"mean_share": imp.mean(axis=1), "min_share": imp.min(axis=1),
                            "max_share": imp.max(axis=1)}).sort_values("mean_share", ascending=False)
imp_summary.round(3)

# %% [markdown]
# ### Backtest: logistic vs. LightGBM vs. ensemble (same rules: weekly, top/bottom 10%, 0.10% per trade)

# %%
bt_models = {"logistic": bt, "lightgbm": backtest(preds_lgb), "ensemble": backtest(ens)}
bt_compare = pd.DataFrame({
    (m, s): perf(b[s]) for m, b in bt_models.items() for s in ["long_short_gross", "long_short_net"]
}).T
bt_compare["avg_turnover"] = [bt_models[m]["turnover"].mean() for m, _ in bt_compare.index]
bt_compare.round(3)

# %%
yearly_net = pd.DataFrame({m: b["long_short_net"].groupby(b.index.year).apply(lambda x: (1 + x).prod() - 1)
                           for m, b in bt_models.items()})
yearly_net.round(3)

# %%
try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    for m in ["logistic", "lightgbm", "ensemble_logit_lgbm", "reversal_1w"]:
        g_ = all_results[all_results["model"] == m]
        ax[0].plot(g_["test_year"], g_["auc"], marker="o", ms=3, label=m)
    ax[0].axhline(0.5, color="grey", lw=0.8); ax[0].set_title("AUC by test year"); ax[0].legend(fontsize=8)
    for m, b in bt_models.items():
        ax[1].plot((1 + b["long_short_net"]).cumprod(), label=f"{m} (net)")
        ax[1].plot((1 + b["long_short_gross"]).cumprod(), ls="--", lw=0.8, label=f"{m} (gross)")
    ax[1].set_yscale("log"); ax[1].set_title("Long-short growth of $1, 2005–2026"); ax[1].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "lightgbm_vs_logistic.png"), dpi=120)
    plt.show()
except ImportError:
    pass

all_results.to_csv(os.path.join(OUT_DIR, "all_models_results_by_year.csv"), index=False)
summary_all.to_csv(os.path.join(OUT_DIR, "all_models_summary.csv"))
preds_lgb.to_csv(os.path.join(OUT_DIR, "oos_predictions_lightgbm.csv.gz"), index=False)
