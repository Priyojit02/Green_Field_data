#!/usr/bin/env python3
import pickle
import re
from difflib import get_close_matches
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------- HARD-CODED PATHS ----------
EXCEL_FILE = r"C:\ML_PRED\Book3.xlsx"  # <-- change if needed
SHEET_NAME = "Sheet1"
OUTPUT_BUNDLE = "model_bundle.pkl"

# ---------- Canonical columns ----------
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

# ---------- Helpers ----------
def clean_header(h) -> str:
    h = str(h)  # ensure string
    h = h.replace("\n", " ")
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

def build_candidates(feature_names):
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

def eval_fixed_split(model, X, y):
    """
    Exactly 4 samples for test (so 9/4 with your 13 rows).
    If y has <13 rows for some target, this still uses 4 test samples
    as long as there are >=5 rows; otherwise it falls back to 25%.
    """
    test_size = 4 if len(X) >= 13 else max(1, int(round(0.25 * len(X))))
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=True
    )
    m = clone(model)
    m.fit(X_tr, y_tr)
    pred = m.predict(X_te)
    r2 = r2_score(y_te, pred) if len(y_te) >= 2 and pd.Series(y_te).nunique() > 1 else np.nan
    mae = mean_absolute_error(y_te, pred)
    return float(r2), float(mae), list(X_te.index)

# ---------- Main ----------
def main():
    print(f"📂 Using Excel: {EXCEL_FILE}")
    print(f"📑 Using sheet: {SHEET_NAME}")

    # headers are on row 3 in your file -> header=2
    df = pd.read_excel(EXCEL_FILE, sheet_name=SHEET_NAME, engine="openpyxl", header=2)
    df = normalize_columns(df)
    df = numericify(df)
    df = df.dropna(how="all")

    if len(df) < 5:
        print("⚠️ Very few rows; scores may be unstable.\n")

    models = {}
    reports = []

    for target in CANONICAL_COLS:
        if target not in df.columns:
            continue

        X = df.drop(columns=[target])
        y = df[target]
        mask = y.notna()
        X, y = X[mask], y[mask]

        # also require at least 2 known features per row
        X = X.dropna(thresh=2)
        y = y.loc[X.index]

        if len(X) < 5:
            print(f"[{target}] ⏭️ skipped (too few usable rows).")
            continue

        candidates = build_candidates(X.columns)

        best = None
        best_name = ""
        best_test_idx = None

        for name, cand in candidates.items():
            r2m, maem, test_idx = eval_fixed_split(cand, X, y)
            print(f"[{target}] {name} → R2={r2m:.3f}  MAE={maem:.3f}")
            if (best is None) or ( (r2m > (best['r2'] if best['r2'] is not None else -1e9)) or
                                   ( (np.isnan(r2m) and np.isnan(best['r2'])) and (maem < best['mae']) ) or
                                   (abs((r2m if not np.isnan(r2m) else 0) - (best['r2'] if not np.isnan(best['r2']) else 0)) < 1e-9 and maem < best['mae']) ):
                best = {"model": cand, "r2": r2m, "mae": maem}
                best_name = name
                best_test_idx = test_idx

        print(f"✅ Best model: {best_name}  (tested rows: {best_test_idx})\n")

        # fit best on all available data for that target
        best["model"].fit(X, y)
        models[target] = best["model"]
        reports.append({
            "target": target,
            "model_name": best_name,
            "r2_mean": best["r2"],
            "mae_mean": best["mae"],
            "r2_std": np.nan,
            "mae_std": np.nan
        })

    bundle = {"models": models, "reports": reports, "columns": CANONICAL_COLS}
    with open(OUTPUT_BUNDLE, "wb") as f:
        pickle.dump(bundle, f)

    print("\n=== ✅ Test-set Scores (9 train / 4 test) ===")
    if reports:
        print(pd.DataFrame(reports).to_string(index=False))
    else:
        print("No targets were trained — please check the Excel data.")
    print(f"\n💾 Saved trained bundle → {OUTPUT_BUNDLE}")

if __name__ == "__main__":
    main()
