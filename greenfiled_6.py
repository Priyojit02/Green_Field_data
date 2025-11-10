#!/usr/bin/env python3
import pickle
import re
from difflib import get_close_matches
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

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

# Targets to log-transform (strictly positive)
LOG_TARGETS = {
    "Client Revenue",
    "Number of Users",
    "RICEFW",
    "Estimated Effort (man days)",
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

def build_preprocessor(feature_names: List[str]) -> ColumnTransformer:
    # Median impute + log1p transform of all features
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("log", FunctionTransformer(np.log1p, validate=False)),
            ]), list(feature_names))
        ],
        remainder="drop"
    )

def build_rf(pre: ColumnTransformer, log_target: bool):
    rf = RandomForestRegressor(
        n_estimators=800,
        max_depth=6,
        min_samples_leaf=2,
        max_features="sqrt",
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
    )
    pipe = Pipeline([("pre", pre), ("rf", rf)])
    if log_target:
        pipe = TransformedTargetRegressor(
            regressor=pipe, func=np.log1p, inverse_func=np.expm1, check_inverse=False
        )
    return pipe

def build_hgb(pre: ColumnTransformer, log_target: bool):
    hgb = HistGradientBoostingRegressor(
        random_state=42,
        learning_rate=0.05,      # conservative for tiny data
        max_leaf_nodes=31,
        min_samples_leaf=2
    )
    pipe = Pipeline([("pre", pre), ("hgb", hgb)])
    if log_target:
        pipe = TransformedTargetRegressor(
            regressor=pipe, func=np.log1p, inverse_func=np.expm1, check_inverse=False
        )
    return pipe

def fixed_split_indices(n_rows: int) -> Tuple[np.ndarray, np.ndarray]:
    # Deterministic split: 9 train / 4 test if possible
    # We’ll let train_test_split decide indices but keep random_state fixed
    test_size = 4 if n_rows >= 13 else min(4, max(1, n_rows // 3))
    return test_size

def evaluate_on_same_split(models: Dict[str, object], X: pd.DataFrame, y: pd.Series, labels: pd.Series):
    """Split once, evaluate all models on the SAME 9/4 test set."""
    test_size = fixed_split_indices(len(X))
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=True
    )

    results = {}
    for name, mdl in models.items():
        m = clone(mdl)
        m.fit(X_tr, y_tr)
        preds = m.predict(X_te)
        r2 = r2_score(y_te, preds) if len(y_te) >= 2 and pd.Series(y_te).nunique() > 1 else np.nan
        mae = mean_absolute_error(y_te, preds)
        results[name] = {
            "r2": float(r2),
            "mae": float(mae),
            "test_clients": labels.loc[X_te.index].astype(str).tolist(),
            "fitted": m
        }
    return results

# ============================
# Main
# ============================
def main():
    print(f"📂 Using Excel: {EXCEL_FILE}")
    print(f"📑 Using sheet: {SHEET_NAME}")

    # Read raw and keep names
    df_raw = pd.read_excel(EXCEL_FILE, sheet_name=SHEET_NAME, engine="openpyxl", header=2)
    if "Greenfield Opportunities" in df_raw.columns:
        labels_full = df_raw["Greenfield Opportunities"].astype(str)
    else:
        labels_full = pd.Series(range(1, len(df_raw) + 1), index=df_raw.index).astype(str)

    # Clean & numeric
    df = normalize_columns(df_raw.copy())
    df = numericify(df)
    df = df.dropna(how="all")
    labels_full = labels_full.loc[df.index]

    models_bundle = {}
    reports = []

    for target in CANONICAL_COLS:
        if target not in df.columns:
            continue

        X = df.drop(columns=[target])
        y = df[target]

        mask = y.notna()
        X, y = X[mask], y[mask]
        labels = labels_full.loc[X.index]

        # Require at least 2 known features
        X = X.dropna(thresh=2)
        y = y.loc[X.index]
        labels = labels.loc[X.index]

        if len(X) < 6:
            print(f"[{target}] ⏭️ skipped (too few usable rows: {len(X)}).\n")
            continue

        pre = build_preprocessor(list(X.columns))
        log_tgt = target in LOG_TARGETS

        candidates = {
            "RandomForest (tiny-data tuned)": build_rf(pre, log_tgt),
            "HistGradientBoosting (tuned)": build_hgb(pre, log_tgt),
        }

        results = evaluate_on_same_split(candidates, X, y, labels)

        # Print both scores (same test set), then choose best
        for name, r in results.items():
            print(f"[{target}] {name} → R2={r['r2']:.3f}  MAE={r['mae']:.3f}")
        # Show test clients once (same split for both)
        some_key = next(iter(results))
        print(f"    test clients: {results[some_key]['test_clients']}")

        # Select best: higher R², tie-breaker lower MAE
        best_name = max(results.keys(), key=lambda k: (np.nan_to_num(results[k]["r2"], nan=-1e9), -results[k]["mae"]))
        best = results[best_name]
        print(f"✅ Best model: {best_name}\n")

        # Refit best on ALL data and store
        best_final = clone(candidates[best_name])
        best_final.fit(X, y)
        models_bundle[target] = best_final

        reports.append({
            "target": target,
            "model_name": best_name + (", log-target" if log_tgt else ""),
            "r2_mean": best["r2"],
            "r2_std": np.nan,   # single split
            "mae_mean": best["mae"],
            "mae_std": np.nan,
        })

    with open(OUTPUT_BUNDLE, "wb") as f:
        pickle.dump({"models": models_bundle, "reports": reports, "columns": CANONICAL_COLS}, f)

    print("\n=== ✅ Fixed Split Scores (9 train / 4 test) ===")
    if reports:
        print(pd.DataFrame(reports).to_string(index=False))
    else:
        print("No targets were trained — please check the data.")
    print(f"\n💾 Saved trained bundle → {OUTPUT_BUNDLE}")

if __name__ == "__main__":
    main()
