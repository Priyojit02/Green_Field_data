#!/usr/bin/env python3
import pickle
import re
from difflib import get_close_matches
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# ============================
# Hardcoded paths (edit here)
# ============================
EXCEL_FILE = r"C:\ML_PRED\Book3.xlsx"
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

# Targets that benefit from log-transform (all are strictly positive in your sheet)
LOG_TARGETS = {
    "Client Revenue",
    "Number of Users",
    "RICEFW",
    "Estimated Effort (man days)",
    # You can add Duration/Countries if you want, but they’re small-range already.
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

def build_rf_pipeline(feature_names: List[str], log_target: bool) -> Pipeline:
    # Log-transform features (log1p) + impute medians
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("log", FunctionTransformer(np.log1p, validate=False)),
            ]), list(feature_names))
        ],
        remainder="drop"
    )

    # Tiny-data tuned RandomForest
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
        # Transform target with log1p for stability; metrics remain in original space
        pipe = TransformedTargetRegressor(
            regressor=pipe,
            func=np.log1p,
            inverse_func=np.expm1,
            check_inverse=False,
        )
    return pipe

def fixed_split_eval(model, X: pd.DataFrame, y: pd.Series, labels: pd.Series):
    """
    Fixed 9/4 split (for 13 usable rows). If fewer rows are available after cleaning,
    it will still try to keep 4 rows for test when possible.
    """
    test_size = 4 if len(X) >= 13 else min(4, max(1, len(X) // 3))
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=True
    )
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)

    # If using TransformedTargetRegressor, pred is already inverse-transformed.
    r2 = r2_score(y_te, pred) if len(y_te) >= 2 and pd.Series(y_te).nunique() > 1 else np.nan
    mae = mean_absolute_error(y_te, pred)

    # Return also the human-friendly client names used in test
    test_names = labels.loc[X_te.index].astype(str).tolist()
    return float(r2), float(mae), test_names, model

# ============================
# Main
# ============================
def main():
    print(f"📂 Using Excel: {EXCEL_FILE}")
    print(f"📑 Using sheet: {SHEET_NAME}")

    # Headers are on row 3 in your file → header=2
    df_raw = pd.read_excel(EXCEL_FILE, sheet_name=SHEET_NAME, engine="openpyxl", header=2)

    # Keep names for reporting
    if "Greenfield Opportunities" in df_raw.columns:
        labels_full = df_raw["Greenfield Opportunities"].astype(str)
    else:
        labels_full = pd.Series(range(1, len(df_raw) + 1), index=df_raw.index).astype(str)

    # Clean & numeric
    df = normalize_columns(df_raw.copy())
    df = numericify(df)
    df = df.dropna(how="all")
    labels_full = labels_full.loc[df.index]

    models = {}
    reports = []

    for target in CANONICAL_COLS:
        if target not in df.columns:
            continue

        # Setup X (other 5 columns) and y (target)
        X = df.drop(columns=[target])
        y = df[target]

        # Keep rows where target exists
        mask = y.notna()
        X, y = X[mask], y[mask]
        labels = labels_full.loc[X.index]

        # Require at least 2 known features
        X = X.dropna(thresh=2)
        y = y.loc[X.index]
        labels = labels.loc[X.index]

        if len(X) < 6:
            print(f"[{target}] ⏭️ skipped (too few usable rows: {len(X)}).")
            continue

        # Build RF pipeline (log-target if appropriate)
        log_tgt = target in LOG_TARGETS
        model = build_rf_pipeline(list(X.columns), log_target=log_tgt)

        # Evaluate with fixed 9/4 split
        r2, mae, test_clients, fitted = fixed_split_eval(model, X, y, labels)
        print(f"[{target}] RandomForest (tiny-data tuned) → R2={r2:.3f}  MAE={mae:.3f}")
        print(f"    test clients: {test_clients}\n")

        # Fit on all available rows afterward (for deployment)
        fitted.fit(X, y)
        models[target] = fitted

        reports.append({
            "target": target,
            "model_name": "RandomForest (tiny-data tuned, log-features" + (", log-target" if log_tgt else "") + ")",
            "r2_mean": r2,
            "r2_std": np.nan,     # single split → no std
            "mae_mean": mae,
            "mae_std": np.nan,
        })

    # Save bundle
    with open(OUTPUT_BUNDLE, "wb") as f:
        pickle.dump({"models": models, "reports": reports, "columns": CANONICAL_COLS}, f)

    print("\n=== ✅ Fixed Split Scores (9 train / 4 test) ===")
    if reports:
        print(pd.DataFrame(reports).to_string(index=False))
    else:
        print("No targets were trained — please check the data.")
    print(f"\n💾 Saved trained bundle → {OUTPUT_BUNDLE}")

if __name__ == "__main__":
    main()
