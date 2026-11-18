"""Features at date t must not change when data after t is removed; labels and folds must be consistent."""
import os
import sys
import numpy as np
import pandas as pd

csv = sys.argv[1]
os.environ["SP500_CSV"] = csv
src = open(os.path.join(os.path.dirname(__file__), "..", "ds675_part1.py")).read()
defs = src.split("# ## Run Steps 1–3")[0]          # definitions only, no execution cells
g = {}
exec(defs, g)

df, _ = g["clean"](g["load_raw"](csv), verbose=False)
full = g["build_panel"](g["to_wide"](df))
ok = True
for cut in ["2003-06-30", "2007-01-15", "2010-11-30"]:
    cut = pd.Timestamp(cut)
    # cleaning uses the next day's price for spike detection, so truncate after cleaning
    trunc = g["build_panel"](g["to_wide"](df[df["Date"] <= cut]), horizon=g["HORIZON"])
    # truncated panel has labels only up to cut-5 days; compare features on overlapping keys
    a = full.set_index(["date", "ticker"])[g["FEATURES"]]
    b = trunc.set_index(["date", "ticker"])[g["FEATURES"]]
    common = a.index.intersection(b.index)
    diff = np.nanmax(np.abs(a.loc[common].to_numpy() - b.loc[common].to_numpy()))
    nan_mismatch = (a.loc[common].isna().to_numpy() != b.loc[common].isna().to_numpy()).sum()
    print(f"cut {cut.date()}: {len(common):,} rows compared, max |diff| = {diff:.2e}, NaN mismatches = {nan_mismatch}")
    ok &= diff < 1e-5 and nan_mismatch == 0

# label check by hand for random rows
w = g["to_wide"](df)
px = w["px"]
for _, r in full.sample(5, random_state=1).iterrows():
    i = r["date_idx"]
    fwd_all = px.iloc[i + 5] / px.iloc[i] - 1
    expect = float(fwd_all[r["ticker"]] > fwd_all.median())
    ok &= expect == r["y"] and np.isclose(fwd_all[r["ticker"]], r["fwd_ret"])

# folds: no train/val label window reaches into the next period, periods disjoint
for f in g["walk_forward_folds"](full):
    tr, va, te = full[f["train"]], full[f["val"]], full[f["test"]]
    ok &= tr["date_idx"].max() + 5 < va["date_idx"].min()
    ok &= va["date_idx"].max() + 5 < te["date_idx"].min()
    ok &= full[f["train_val"]]["date_idx"].max() + 5 < te["date_idx"].min()
print("ALL CHECKS PASSED" if ok else "CHECK FAILED")
sys.exit(0 if ok else 1)
