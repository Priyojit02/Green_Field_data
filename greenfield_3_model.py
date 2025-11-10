#!/usr/bin/env python3
import pickle
import re
from collections import Counter, defaultdict
from difflib import get_close_matches
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

# ============================
# Hardcoded paths (edit here)
# ============================
EXCEL_FILE = r"C:\ML_PRED\Book3.xlsx"   # <-- change if needed
SHEET_NAME = "Sheet1"
OUTPUT_BUNDLE = "model_bundle.pkl"

# ============================
# Canonical columns & aliases
# ============================
CANONICAL_COLS = [
    "Client Revenue",
    "Number of Users",
    "RICEFW",
    "Duration (Months)",
    "Countries/Market",
    "Estimated Effort (man days)",
]

ALIASES = {
    r"^client\s*revenue$": "Client Revenue",
    r"^revenue$": "Client Revenue",
    r"^number\s*of\s*users$": "Number of Users",
    r"^users$": "Number of Users",
    r"^ricefw$": "RICEFW",
    r"^duration(\s*\(months\))?$": "Duration (Months)",
    r"^countries\s*/\s*market$": "Countries/Market",
    r"^countries$": "Countries/Market",
    r"^estimated\s*effort(\s*\(man\s*days\))?$": "Estimated Effort (man days)",
    r"^effort$": "Estimated Effort (man days)",
}

# ============================
# Helpers
# ============================
def clean_header(h) -> str:
    h = str(h).replace("\n", " ")
    return re.sub(r"\s+", " ", h.strip()).replace("’", "'")

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    original = list(df.columns)
    rename_map: Dict[str, str] = {}
    for col in df.columns:
        key = clean_header(col).lower()
        mapped = None
        for patt, canon in ALIASES.items():
            if re.fullmatch(patt, key):
                mapped = canon
                break
        if mapped is None:
            close = get_close_matches(key, [c.lower() for c in CANONICAL_COLS], n=1, cutoff=0.8)
            if close:
                mapped = CANONICAL_COLS[[c.lower() for c in CANONICAL_COLS].index(close[0])]
        if mapped is None:
            mapped = clean_header(col)
        rename_map[col] = mapped

    df = df.rename(columns=rename_map)
    print("🧾 Original headers:", original)
    print("✅ Normalized headers:", list(df.columns))
    keep = [c for c in CANONICAL_COLS if c in df.columns]
    print("📊 Keeping columns:", keep)
    return df[keep]

def numericify(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        df[c] = (
            df[c].astype(str)
                 .str.replace(",", "", regex=False)
                 .str.replace(" ", "", regex=False)
        )
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def build_candidates(feature_names: List[str]):
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    pre = ColumnTransformer([("num", numeric, list(feature_names))], remainder="drop")

    hgb = HistGradientBoostingRegressor(
        random_state=42, learning_rate=0.07, max_leaf_nodes=31, min_samples_leaf=2
    )
    rf = RandomForestRegressor(n_estimators=600, random_state=42, n_jobs=-1)

    return {
        "HistGradientBoosting": Pipeline([("pre", pre), ("reg", hgb)]),
        "RandomForest": Pipeline([("pre", pre), ("reg", rf)]),
    }

# ============================
# Bootstrap evaluation (tracks who & how often is tested)
# ============================
def bootstrap_eval(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    labels: pd.Series,
    n_bootstraps: int = 300,
    verbose_iters: int = 3
) -> Tuple[float, float, float, float, Counter, Counter, List[List[str]]]:
    """
    Runs bootstrap resampling and returns:
      r2_mean, r2_std, mae_mean, mae_std,
      test_counts (Counter of index -> #times used as test),
      test_size_hist (Counter of test set sizes),
      samples_preview (list of label lists for first `verbose_iters` iterations).
    """
    r2s, maes = [], []
    test_counts = Counter()
    test_size_hist = Counter()
    samples_preview: List[List[str]] = []

    n = len(X)
    for b in range(n_bootstraps):
        # Bootstrap training set (with replacement)
        X_train, y_train = resample(X, y, n_samples=n, replace=True, random_state=None)
        # Out-of-bag = test set
        mask = ~X.index.isin(X_train.index)
        X_test, y_test = X.loc[mask], y.loc[mask]

        test_size = len(X_test)
        test_size_hist[test_size] += 1
        test_counts.update(X_test.index.tolist())

        if b < verbose_iters:
            samples_preview.append(labels.loc[X_test.index].astype(str).tolist())

        # Need at least 2 test samples and >1 unique target value to compute R²
        if test_size < 2 or pd.Series(y_test).nunique() < 2:
            continue

        m = clone(model)
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        r2s.append(r2_score(y_test, preds))
        maes.append(mean_absolute_error(y_test, preds))

    r2_mean = float(np.mean(r2s)) if r2s else np.nan
    r2_std  = float(np.std(r2s))  if r2s else np.nan
    mae_mean = float(np.mean(maes)) if maes else np.nan
    mae_std  = float(np.std(maes))  if maes else np.nan

    return r2_mean, r2_std, mae_mean, mae_std, test_counts, test_size_hist, samples_preview

# ============================
# Main
# ============================
def main():
    print(f"📂 Using Excel: {EXCEL_FILE}")
    print(f"📑 Using sheet: {SHEET_NAME}")

    # Headers in your sheet are on row 3 → header=2
    df_raw = pd.read_excel(EXCEL_FILE, sheet_name=SHEET_NAME, engine="openpyxl", header=2)

    # Keep client names/IDs for reporting (if available)
    if "Greenfield Opportunities" in df_raw.columns:
        labels_full = df_raw["Greenfield Opportunities"].astype(str)
    else:
        labels_full = pd.Series(range(1, len(df_raw) + 1), index=df_raw.index, name="ID").astype(str)

    # Normalize, numericify, align labels with cleaned df
    df = normalize_columns(df_raw.copy())
    df = numericify(df)
    df = df.dropna(how="all")
    labels_full = labels_full.loc[df.index]

    models: Dict[str, Pipeline] = {}
    reports: List[Dict] = []

    for target in CANONICAL_COLS:
        if target not in df.columns:
            continue

        X = df.drop(columns=[target])
        y = df[target]

        # Only rows where target is known
        mask = y.notna()
        X, y = X[mask], y[mask]
        labels = labels_full.loc[X.index]

        # Also require at least 2 known features per row for training
        X = X.dropna(thresh=2)
        y = y.loc[X.index]
        labels = labels.loc[X.index]

        if len(X) < 6:
            print(f"[{target}] ⏭️ skipped (too few usable rows: {len(X)}).")
            continue

        candidates = build_candidates(list(X.columns))

        best = None
        best_name = ""
        best_counts = None
        best_size_hist = None
        best_preview = None

        for name, cand in candidates.items():
            r2m, r2s, maem, maes, counts, size_hist, preview = bootstrap_eval(
                cand, X, y, labels, n_bootstraps=300, verbose_iters=3
            )
            print(f"[{target}] {name} → R2={r2m:.3f}±{r2s:.3f}  MAE={maem:.3f}±{maes:.3f}")

            if (best is None) or (r2m > best["r2"]) or (abs(r2m - best["r2"]) < 1e-6 and maem < best["mae"]):
                best = {"model": cand, "r2": r2m, "r2s": r2s, "mae": maem, "maes": maes}
                best_name = name
                best_counts = counts
                best_size_hist = size_hist
                best_preview = preview

        print(f"✅ Best model: {best_name} (R2={best['r2']:.3f}±{best['r2s']:.3f}, MAE={best['mae']:.3f}±{best['maes']:.3f})")

        # Show exactly who was in test for the first few bootstrap iterations
        if best_preview:
            for i, names in enumerate(best_preview, 1):
                print(f"   Iter {i} test set: {names}")

        # Show which clients were most often in test across all bootstraps
        if best_counts:
            print("   🔎 Most-tested clients (top 10):")
            # best_counts is Counter of indices; map to names
            top10 = best_counts.most_common(10)
            for idx, cnt in top10:
                name = labels.loc[idx] if idx in labels.index else str(idx)
                print(f"      {name}: {cnt} times")

        # Show distribution of test-set sizes (so you see 9/4 vs 10/3 vs 8/5, etc.)
        if best_size_hist:
            print("   🧮 Test size frequency:")
            for sz in sorted(best_size_hist):
                print(f"      {sz} rows in test → {best_size_hist[sz]} runs")

        print()  # spacer

        # Fit best model on all available rows (for deployment)
        best["model"].fit(X, y)
        models[target] = best["model"]
        reports.append({
            "target": target,
            "model_name": best_name,
            "r2_mean": best["r2"],
            "r2_std": best["r2s"],
            "mae_mean": best["mae"],
            "mae_std": best["maes"],
        })

    # Save bundle
    with open(OUTPUT_BUNDLE, "wb") as f:
        pickle.dump(
            {"models": models, "reports": reports, "columns": CANONICAL_COLS},
            f
        )

    print("\n=== ✅ Bootstrap Scores (mean ± std over 300 runs) ===")
    if reports:
        print(pd.DataFrame(reports).to_string(index=False))
    else:
        print("No targets were trained — please check the data.")
    print(f"\n💾 Saved trained bundle → {OUTPUT_BUNDLE}")

if __name__ == "__main__":
    main()
