# DS675 Mini Project — Relative Stock Returns on S&P 500 Data

**Group:** Eapen Abraham, Ketan Verma, Pothuri Aditya, Veerasingam Agileswaran

**Question:** over the next 5 trading days, will a given stock do better or worse than the median S&P 500 stock?

Each stock on each day gets the label `1` if its 5-day forward return (Adj Close) is above that day's
median across all stocks, else `0`. About half the stocks are `1` every day, so random guessing scores 50%
and anything reliably above that is real skill.

## What's in this repo

| File | What it does |
|---|---|
| `ds675_part1.ipynb` | Part 1 notebook: cleaning, label, features, walk-forward split, baselines, evaluation |
| `ds675_part1.py` | The same code as a plain Python script (paired with the notebook via jupytext) |
| `tests/test_no_lookahead.py` | Checks that features use no future data, labels are correct and folds don't overlap |
| `tests/make_synthetic.py` | Builds a fake dataset with the same columns and quirks, for testing without the real file |
| `requirements.txt` | Python packages |

Part 1 covers Milestone 2 Steps 1–3, 5, 6 and 8:

1. **Clean** — Adj Close for all returns, zero prices → missing, duplicates dropped, single-day bad prints removed, `CICC` excluded, data from 2000 on (plus 400 days of warm-up history).
2. **Label** — 5-day forward return vs. that day's cross-sectional median.
3. **Features** — 16 features from past data only (returns over 1/5/21/63 days, 12-1 momentum, volatility, volume ratio, dollar volume, RSI, MACD, Bollinger %B, distance from 52-week high), each converted to a rank among all stocks that day (0 = lowest, 1 = highest).
5. **Walk-forward split** — train on years ≤ Y, tune on Y+1, test on Y+2, same cutoffs for every stock, last 5 days before each period purged. 22 test years (2005–2026).
6. **Baselines** — random, 12-1 momentum, 1-month momentum, 1-week reversal, logistic regression.
8. **Metrics** — accuracy, AUC, daily rank information coefficient (IC) with t-stat, top-minus-bottom decile 5-day return spread.

Still to come: peer-group clustering (Step 4), LightGBM and LSTM (Step 7), portfolio simulation (Step 9).

## 1. Get the data

The dataset is **not** stored in this repo (it's ~255 MB). It's on Kaggle:
<https://www.kaggle.com/datasets/parsalatifi/s-and-p-500-us-stock-market-data>

You need a free Kaggle account. Pick one option:

**A. Website:** click **Download**, unzip, and put the CSV in a `data/` folder in this repo.

**B. Kaggle CLI:** create an API token at Kaggle → Settings → API → *Create New Token*, save `kaggle.json` to
`~/.kaggle/` (Windows: `C:\Users\<you>\.kaggle\`), then:

```bash
pip install kaggle
kaggle datasets download -d parsalatifi/s-and-p-500-us-stock-market-data -p data --unzip
```

**C. Run on Kaggle itself:** see [Option 3](#option-3--kaggle-notebook-no-download) below.

`data/` is in `.gitignore`, so the CSV will never be committed by accident.

## 2. Set up Python

Python 3.10 or newer.

```bash
git clone https://github.com/ap3478/CS675-Project.git
cd CS675-Project
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Run it

The code finds the CSV automatically: first the `SP500_CSV` environment variable, then `/kaggle/input/`,
then `./` and `./data/` (it picks the largest CSV). Outputs go to `OUT_DIR` (default: current folder).

### Option 1 — Jupyter notebook (best for exploring and for the report)

```bash
jupyter lab ds675_part1.ipynb      # or: jupyter notebook ds675_part1.ipynb
```

Then **Run All**. Tables and charts appear under each cell, and you can rerun single cells (e.g. just the
baselines) without reloading the data. To use a CSV outside `./data/`, set `SP500_CSV` before starting Jupyter.

### Option 2 — Plain Python script (best for full, reproducible runs)

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

A script only prints what the code explicitly prints (the cleaning report and progress per fold); the
summary tables are saved to files instead — see [Outputs](#outputs). `MPLBACKEND=Agg` saves the chart
without opening a window that would pause the script.

### Option 3 — Kaggle notebook (no download)

1. Open the dataset page on Kaggle and click **New Notebook** (the data is attached under `/kaggle/input/`).
2. **File → Import Notebook** and upload `ds675_part1.ipynb`.
3. **Run All**. Download the result files from the notebook's output panel.

### Runtime

On a full-size test (233 tickers, 2000–2026, ~1.4 M labelled rows): about 80 seconds and 1.2 GB of RAM on a
laptop-class CPU. No GPU needed for Part 1.

## Outputs

| File | Contents |
|---|---|
| `baseline_summary.csv` | One row per model: mean accuracy, AUC, IC, IC t-stat across years, years with positive IC, decile spread |
| `baseline_results_by_year.csv` | Every metric for every model and test year |
| `baselines_by_year.png` | AUC and IC by test year for each model |
| `panel.parquet` | Only if `SAVE_PANEL=1`: the full labelled feature table, for reuse by later steps (needs `pyarrow`) |

How to read them: random should sit at accuracy/AUC ≈ 0.50 and IC ≈ 0. A model is useful if its AUC is
consistently above 0.50 and its IC is positive in most test years.

## Configuration

Edit the **Configuration** cell at the top of the notebook/script:

| Setting | Default | Meaning |
|---|---|---|
| `START_DATE` | 2000-01-01 | First date that gets a label |
| `HORIZON` | 5 | Forward return horizon in trading days |
| `EXCLUDE_TICKERS` | `["CICC"]` | Tickers removed outright |
| `FLAG_ABS_DAILY_RET` | 1.0 | Tickers with a one-day move above 100% are *listed* for review (not dropped) |
| `MIN_TRAIN_YEARS` | 4 | First fold trains on 2000–2003, tunes on 2004, tests on 2005 |
| `LOGIT_C_GRID` | 0.001–1 | Regularisation values tried for logistic regression |
| `REFIT_ON_TRAIN_VAL` | `True` | After choosing C on the validation year, refit on train + validation. Set `False` to train on years ≤ Y only |

After the first run on real data, check the list of flagged tickers printed after cleaning and add any
that look broken to `EXCLUDE_TICKERS`.

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

You can also point `test_no_lookahead.py` at the real CSV (slower, ~1–2 minutes).

To check the whole pipeline end to end on synthetic data:

```bash
SP500_CSV=data/synthetic.csv OUT_DIR=results MPLBACKEND=Agg python ds675_part1.py
```

The synthetic data has a weak 1-week reversal effect planted in it, so `reversal_1w` and `logistic` should
beat 0.50 AUC, while `random` and `momentum_12_1` should stay at about 0.50.

## Editing: keep the notebook and script in sync

`ds675_part1.py` uses jupytext's percent format (`# %%` marks a cell). After editing either file:

```bash
jupytext --sync ds675_part1.py      # or: jupytext --sync ds675_part1.ipynb
```

Prefer editing the `.py` and committing both; notebook diffs are hard to merge. Clear notebook outputs before
committing (*Kernel → Restart & Clear Output*) to keep diffs small.

## Data notes

* Survivorship bias: the dataset contains only current S&P 500 members (233 of 250 requested tickers), so
  long–short results will be optimistic. Treat backtest returns as an upper bound.
* The panel is unbalanced: about 130 stocks per day in 2000 vs. about 230 today. Ranks and the median-based
  label adjust to the number of stocks each day.
