# DS675 Mini Project — Relative Stock Returns on S&P 500 Data

**Group:** Eapen Abraham, Ketan Verma, Pothuri Aditya, Veerasingam Agileswaran

**Question:** over the next 5 trading days, will a given stock do better or worse than the median S&P 500 stock?

Each stock on each day gets the label `1` if its 5-day forward return (Adj Close) is above that day's
median across all stocks, else `0`. About half the stocks are `1` every day, so random guessing scores 50%
and anything reliably above that is real skill.

## Results so far (Kaggle run, 23 November 2026)

| Model | AUC | Mean IC | Years IC > 0 | Long-short after costs |
|---|---|---|---|---|
| Ensemble (logistic + LightGBM) | 0.5159 | 0.0328 | 22/22 | 4.1% / year |
| LightGBM | 0.5156 | 0.0315 | 22/22 | 4.5% / year |
| Logistic regression | 0.5149 | 0.0300 | 22/22 | 2.6% / year |
| 1-week reversal | 0.5090 | 0.0199 | 20/22 | — |
| Random | 0.4993 | −0.0021 | 7/22 | — |

Small but consistent skill in every out-of-sample year 2005–2026; trading costs (0.10% per dollar traded,
about 63% weekly turnover) take most of the profit. The full run with all outputs is in
[`runs/ds675_part1_kaggle_run_2026-11-23.ipynb`](runs/ds675_part1_kaggle_run_2026-11-23.ipynb).

## What's in this repo

| File | What it does |
|---|---|
| `ds675_part1.ipynb` | The main notebook (73 cells, no outputs): every analysis below |
| `ds675_part1.py` | The same code as a plain Python script, paired with the notebook via jupytext |
| `runs/ds675_part1_kaggle_run_2026-11-23.ipynb` | The notebook as run on Kaggle on the real data, **with all outputs** |
| `tests/test_no_lookahead.py` | Checks that features use no future data, labels are correct and folds don't overlap |
| `tests/make_synthetic.py` | Builds a fake dataset with the same columns and quirks, for testing without the real file |
| `requirements.txt` | Python packages |

### Sections of the notebook

| Section | Milestone step | What it does |
|---|---|---|
| Configuration | — | Settings (dates, horizon, excluded tickers, model options) |
| Step 1 — Load and clean | 1 | Adj Close for all returns, zero prices → missing, duplicates dropped, single-day bad prints removed, `CICC` excluded, data from 2000 (plus warm-up); lists tickers with one-day moves > 100% for review |
| Step 2 — Label | 2 | 5-day forward return vs. that day's cross-sectional median |
| Step 3 — Features | 3 | 16 features from past prices only (returns, momentum, volatility, volume, size, RSI, MACD, Bollinger %B, 52-week high), each ranked among all stocks that day |
| Step 5 — Walk-forward folds | 5 | Train on years ≤ Y, tune on Y+1, test on Y+2; same cutoffs for every stock; 5-day purge. 22 test years (2005–2026) |
| Steps 6, 8 — Baselines and metrics | 6, 8 | Random, 12-1 momentum, 1-month momentum, 1-week reversal, logistic regression; accuracy, AUC, rank IC, IC t-stat, top-minus-bottom decile spread |
| Predicting the past | 8 | Saves every out-of-sample prediction (`oos_predictions.csv.gz`) |
| Look up one stock | — | Predictions vs. what happened for any ticker and period, label check against the median stock, hit rate by model confidence, best/worst 10 stocks, accuracy by company size |
| Backtest | 9 | Weekly long-short top/bottom 10%, 0.10% cost per dollar traded; long-only and all-stocks comparisons; yearly returns and growth chart |
| PCA | — | PCA on the 16 features (variance, loadings, models on k components, refit inside each fold) and on daily returns (market factor, sector-like components, nearest peers of a stock, market factor by year) |
| Step 7a — LightGBM | 7 | Gradient-boosted trees on the same folds with early stopping; ensemble with logistic regression; feature importance; backtest comparison |

Still to come: peer-group clustering (Step 4), the LSTM (Step 7b).

## 1. Get the data

The dataset is **not** stored in this repo (~255 MB). It's on Kaggle:
<https://www.kaggle.com/datasets/parsalatifi/s-and-p-500-us-stock-market-data>

You need a free Kaggle account. Pick one option:

**A. Website:** click **Download**, unzip, and put the CSV in a `data/` folder in this repo.

**B. Kaggle CLI:** create an API token at Kaggle → Settings → API → *Create New Token*, save `kaggle.json` to
`~/.kaggle/` (Windows: `C:\Users\<you>\.kaggle\`), then:

```bash
pip install kaggle
kaggle datasets download -d parsalatifi/s-and-p-500-us-stock-market-data -p data --unzip
```

**C. Run on Kaggle itself:** see [Option 1](#option-1--kaggle-notebook-recommended) below; no download needed.

`data/` is in `.gitignore`, so the CSV will never be committed by accident.

## 2. Run it

The code finds the CSV automatically: first the `SP500_CSV` environment variable, then `/kaggle/input/`,
then `./` and `./data/` (it picks the largest CSV). Outputs go to `OUT_DIR` (default: current folder;
`/kaggle/working` on Kaggle).

**Runtime** for the whole notebook on the real data: about 10 minutes on Kaggle's CPU (loading and features
~1 min, baselines ~2.5 min, predictions ~0.5 min, PCA ~1 min, LightGBM ~3 min), well within Kaggle's memory.
No GPU needed.

### Option 1 — Kaggle notebook (recommended)

1. Open the dataset page on Kaggle and click **New Notebook** (the data is attached under `/kaggle/input/`).
2. **File → Import Notebook** and upload `ds675_part1.ipynb` from this repo.
3. Click **Save Version → Save & Run All (Commit)**. This runs the whole notebook in the background and
   **keeps all outputs**; outputs from an interactive session are lost when the session ends.
4. When the version finishes, open it. Download result files from its **Output** tab, or the notebook with
   outputs from the **⋮** menu → **Download**.

If you run cells interactively instead (**Run All**), every variable lives in memory and disappears when the
session stops or times out. A `NameError: name 'preds' is not defined` means the session restarted: choose
**Run → Run All** again, or load saved predictions (see [Looking up from saved predictions](#looking-up-from-saved-predictions-no-rerun)).

### Option 2 — Jupyter on your own computer

Python 3.10 or newer.

```bash
git clone https://github.com/ap3478/CS675-Project.git
cd CS675-Project
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab ds675_part1.ipynb      # then Run All
```

To use a CSV outside `./data/`, set `SP500_CSV` before starting Jupyter.

### Option 3 — Plain Python script

macOS / Linux:
```bash
mkdir -p results
SP500_CSV=data/<file>.csv OUT_DIR=results MPLBACKEND=Agg python ds675_part1.py
```

Windows PowerShell:
```powershell
mkdir results
$env:SP500_CSV="data\<file>.csv"; $env:OUT_DIR="results"; $env:MPLBACKEND="Agg"
python ds675_part1.py
```

A script only shows what the code explicitly prints (cleaning report, progress per fold, a few summaries);
the tables are saved to files instead (see [Outputs](#outputs)). `MPLBACKEND=Agg` saves charts without opening
windows that would pause the script.

## 3. Look up a different stock (change the ticker)

The lookup section of the notebook starts with a cell under the heading **"Look up one stock"**. Its first line is:

```python
TICKER, FROM, TO = "AAPL", "2020-02-15", "2022-04-15"   # change the ticker and dates here
```

1. Replace `"AAPL"` with the ticker you want, **in quotes and capitals**, e.g. `"NVDA"`, `"JPM"`, `"XOM"`.
   Change `FROM` and `TO` to the period you want (format `"YYYY-MM-DD"`).
   Predictions exist from **2005-01-03 to 2026-08-21**; a range outside that returns an empty table.
2. Run that cell, then the next two cells:
   - **"Check the labels for the looked-up stock against the median stock"**: the stock's 5-day return next to
     the median stock's, the hit rate for the period, and `labels match recomputation: True`.
   - **"Is the model more accurate when it is more confident about this stock?"**: hit rate by how strongly the
     model ranked the stock, and the stock's full 2005–2026 hit rate vs. the average stock.

You don't need to rerun the whole notebook, as long as the session is still running (`preds` exists).

Tips:
- **Only replace the ticker value.** Don't edit `preds["ticker"]` elsewhere in the cell: `preds["AAPL"]`
  raises `KeyError: 'AAPL'` because there is no column with that name.
- **See every ticker available:** run `sorted(preds["ticker"].unique())` in a new cell (231 tickers).
  A ticker that isn't in the data prints a message and falls back to the first ticker.
- **Show every row of a long period:** add `pd.set_option("display.max_rows", 1000)` at the top of the cell.
- **One day's top and bottom picks:** in the cell below the lookup, set e.g. `DAY = pd.Timestamp("2020-03-16")`
  (must be a trading day).

### Looking up from saved predictions (no rerun)

After a **Save Version** run, the predictions are saved as `oos_predictions.csv.gz` (logistic) and
`oos_predictions_lightgbm.csv.gz` (LightGBM). On Kaggle, attach that run's output to a notebook with
**+ Add Input → Your Work** and run:

```python
import glob
import pandas as pd

path = glob.glob("/kaggle/input/**/oos_predictions.csv.gz", recursive=True)[0]   # or a local path
preds = pd.read_csv(path, parse_dates=["date"])

TICKER, FROM, TO = "AAPL", "2020-02-15", "2022-04-15"
a = preds[(preds["ticker"] == TICKER) & preds["date"].between(FROM, TO)].copy()
med = preds.groupby("date")["fwd_ret"].median()
a["median_stock_5d"] = a["date"].map(med)
print(f"{TICKER}: {len(a)} days, model right {a['correct'].mean():.1%}")
a[["date", "rank_pct", "predicted_y", "fwd_ret", "median_stock_5d", "y", "correct"]]
```

Columns: `prob_beat_median` (model probability), `rank_pct` (rank that day, 1 = model's favourite),
`predicted_y` (model's call), `y` (1 = actually beat the median), `fwd_ret` (actual 5-day return),
`correct` (call was right).

## Outputs

| File | Contents |
|---|---|
| `baseline_summary.csv` | One row per baseline: accuracy, AUC, IC, IC t-stat across years, years with positive IC, decile spread |
| `baseline_results_by_year.csv` | Every metric for every baseline and test year |
| `oos_predictions.csv.gz` | Every out-of-sample logistic prediction (about 1.18 million rows) |
| `backtest_weekly.csv` | Weekly long-short, long-only and all-stocks returns, and turnover |
| `pca_logit_results_by_year.csv` | Models on k PCA components, by test year |
| `all_models_summary.csv` / `all_models_results_by_year.csv` | All models including LightGBM and the ensemble |
| `oos_predictions_lightgbm.csv.gz` | Every out-of-sample LightGBM prediction |
| `baselines_by_year.png`, `backtest_growth.png`, `pca_features.png`, `pca_returns.png`, `lightgbm_vs_logistic.png` | Charts |
| `panel.parquet` | Only if `SAVE_PANEL=1`: the full labelled feature table (needs `pyarrow`) |

How to read them: random should sit at accuracy/AUC ≈ 0.50 and IC ≈ 0. A model is useful if its AUC is
consistently above 0.50 and its IC is positive in most test years. AUC above about 0.58 would suggest leakage,
not skill.

## Configuration

Edit the **Configuration** cell near the top of the notebook/script, or the setting cells in each section:

| Setting | Default | Meaning |
|---|---|---|
| `START_DATE` | 2000-01-01 | First date that gets a label |
| `HORIZON` | 5 | Forward return horizon in trading days |
| `EXCLUDE_TICKERS` | `["CICC"]` | Tickers removed outright |
| `FLAG_ABS_DAILY_RET` | 1.0 | Tickers with a one-day move above 100% are *listed* for review (not dropped) |
| `MIN_TRAIN_YEARS` | 4 | First fold trains on 2000–2003, tunes on 2004, tests on 2005 |
| `LOGIT_C_GRID` | 0.001–1 | Regularisation values tried for logistic regression |
| `REFIT_ON_TRAIN_VAL` | `True` | After tuning on the validation year, refit on train + validation |
| `TICKER, FROM, TO` | AAPL, 2020-02-15, 2022-04-15 | Stock and period for the lookup ([section 3](#3-look-up-a-different-stock-change-the-ticker)) |
| `COST_PER_TRADE` | 0.001 | Backtest cost per dollar traded (0.10%) |
| `DECILE` | 0.10 | Backtest buys the top and shorts the bottom 10% |
| `PCA_K_LIST` | 1, 2, 3, 5, 8, 16 | Numbers of PCA components tested |
| `PEER_OF` | AAPL | Stock whose nearest return peers are listed in the PCA section |
| `LGB_PARAMS` | see notebook | LightGBM settings (learning rate 0.05, ≥ 2,000 rows per leaf, subsampling, L2) |

After the first run on new data, check the list of flagged tickers printed after cleaning and add any
that look broken to `EXCLUDE_TICKERS`. In the 2026 run only HIG and WMB were flagged, both real events.

## Tests

The tests run on the synthetic dataset, so they work without downloading anything:

```bash
python tests/make_synthetic.py data/synthetic.csv 120 1998-01-01 2012-12-31
python tests/test_no_lookahead.py data/synthetic.csv
```

Expected last line: `ALL CHECKS PASSED`. The test

* rebuilds the features on data cut off at three dates and checks they are identical to the full-data features (no lookahead),
* recomputes five random labels by hand,
* checks that no training or validation label window reaches into the following period.

To check the whole notebook end to end on synthetic data (about a minute):

```bash
SP500_CSV=data/synthetic.csv OUT_DIR=results MPLBACKEND=Agg python ds675_part1.py
```

The synthetic data has a weak 1-week reversal effect planted in it, so `reversal_1w`, `logistic` and
`lightgbm` should beat 0.50 AUC (LightGBM should put most of its importance on `ret_5`), while `random` and
`momentum_12_1` stay at about 0.50. The synthetic data has no AAPL and ends in 2012, so the lookup falls back
to the first ticker and shows an empty table for the default 2020–2022 period.

## Editing: keep the notebook and script in sync

`ds675_part1.py` uses jupytext's percent format (`# %%` marks a cell). After editing either file:

```bash
jupytext --sync ds675_part1.ipynb
```

Prefer editing the `.py` and committing both; notebook diffs are hard to merge. Keep `ds675_part1.ipynb`
without outputs (*Kernel → Restart & Clear Output* before committing). Runs with outputs you want to keep go
in `runs/`, named with the date.

## Data notes

* **Survivorship bias:** the dataset contains only current S&P 500 members (233 of 250 requested tickers).
  The equal-weight average stock returned 16.9% a year in 2005–2026, well above the real index, and models
  learn that small, volatile stocks did well. Treat long-only results as an upper bound; long-short is the
  fairer test.
* **Unbalanced panel:** 175 stocks per day in 2000 vs. 231 today. Ranks and the median-based label adjust to
  the number of stocks each day.
* **2026 is a partial year** (data ends 31 August 2026; the last label is 21 August).
