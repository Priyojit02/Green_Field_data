#!/usr/bin/env python3
import argparse
import pickle
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CANONICAL_COLS = [
    "Client Revenue",
    "Number of Users",
    "RICEFW",
    "Duration (Months)",
    "Countries/Market",
    "Estimated Effort (man days)",
]

ALT_NAMES = {
    "client revenue": "Client Revenue",
    "revenue": "Client Revenue",
    "number of users": "Number of Users",
    "users": "Number of Users",
    "ricefw": "RICEFW",
    "duration (months)": "Duration (Months)",
    "duration": "Duration (Months)",
    "countries/ market": "Countries/Market",
    "countries/market": "Countries/Market",
    "countries": "Countries/Market",
    "estimated effort (man days)": "Estimated Effort (man days)",
    "estimated effort": "Estimated Effort (man days)",
    "effort": "Estimated Effort (man days)",
}

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for col in df.columns:
        key = col.strip().lower()
        mapping[col] = ALT_NAMES.get(key, col.strip())
    df = df.rename(columns=mapping)
    keep = [c for c in CANONICAL_COLS if c in df.columns]
    return df[keep]

def robust_to_numeric(s: pd.Series) -> pd.Series:
    s = (s.astype(str)
          .str.replace(",", "", regex=False)
          .str.replace("₹", "", regex=False)
          .str.replace("$", "", regex=False)
          .str.replace("€", "", regex=False)
          .str.replace("£", "", regex=False)
          .str.replace("B", "", regex=False)  # allow entries like '34B'
          .str.strip())
    return pd.to_numeric(s, errors="coerce")

@dataclass
class ModelReport:
    target: str
    model_name: str
    r2_mean: float
    mae_mean: float
    r2_std: float
    mae_std: float

def loocv_scores(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> Tuple[float, float, float, float]:
    n = len(X)
    kf = KFold(n_splits=n, shuffle=True, random_state=42)
    r2s, maes = [], []
    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        r2s.append(r2_score(y_te, pred))
        maes.append(mean_absolute_error(y_te, pred))
    return float(np.mean(r2s)), float(np.mean(maes)), float(np.std(r2s)), float(np.std(maes))

def build_candidates(random_state=42) -> Dict[str, Pipeline]:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    pre = ColumnTransformer([("num", numeric, list(range(0, 10)))], remainder="drop")

    hgb = HistGradientBoostingRegressor(
        max_depth=None, learning_rate=0.07, max_leaf_nodes=31,
        min_samples_leaf=2, l2_regularization=0.0, random_state=random_state
    )
    rf = RandomForestRegressor(
        n_estimators=600, max_depth=None, min_samples_leaf=1,
        random_state=random_state, n_jobs=-1
    )

    return {
        "HistGradientBoosting": Pipeline([("pre", pre), ("reg", hgb)]),
        "RandomForest": Pipeline([("pre", pre), ("reg", rf)]),
    }

def auto_find_excel(path: str=".") -> Optional[str]:
    from glob import glob
    cands = glob(f"{path}/*.xlsx") + glob(f"{path}/*.xls")
    return cands[0] if cands else None

def pick_best_sheet(xl_path: str) -> str:
    xls = pd.ExcelFile(xl_path)
    best_sheet, best_score = None, -1
    for sheet in xls.sheet_names:
        df = pd.read_excel(xl_path, sheet_name=sheet, nrows=50)
        df_norm = normalize_columns(df)
        score = len(df_norm.columns.intersection(CANONICAL_COLS))
        if score > best_score:
            best_sheet, best_score = sheet, score
    if best_sheet is None:
        raise SystemExit("No suitable sheet found.")
    return best_sheet

def train_ompt(df: pd.DataFrame, verbose: bool=True):
    models = {}
    reports: List[ModelReport] = []
    for target in CANONICAL_COLS:
        if target not in df.columns:
            continue
        X = df.drop(columns=[target])
        y = df[target]
        candidates = build_candidates()

        best_model, best_r2, best_mae, rep_chosen = None, -1e9, 1e9, None
        for name, model in candidates.items():
            r2m, maem, r2s, maes = loocv_scores(model, X, y)
            if verbose:
                print(f"[{target}] {name}: R2={r2m:.3f} ± {r2s:.3f} | MAE={maem:.3f} ± {maes:.3f}")
            if (r2m > best_r2) or (abs(r2m - best_r2) < 1e-6 and maem < best_mae):
                best_r2, best_mae = r2m, maem
                best_model = model
                rep_chosen = ModelReport(target, name, r2m, maem, r2s, maes)

        best_model.fit(X, y)
        models[target] = best_model
        reports.append(rep_chosen)

    return models, reports

def main():
    ap = argparse.ArgumentParser(description="Dynamic trainer for Greenfield models (auto-finds Excel & sheet).")
    ap.add_argument("--excel", default=None, help="Path to Excel file. If omitted, searches current folder.")
    ap.add_argument("--sheet", default=None, help="Sheet name/index. If omitted, best sheet is auto-picked.")
    ap.add_argument("--bundle", default="model_bundle.pkl", help="Output pickle bundle.")
    args = ap.parse_args()

    xl_path = args.excel or auto_find_excel(".")
    if not xl_path:
        raise SystemExit("No Excel file found. Place a .xlsx/.xls here or pass --excel path.")
    print(f"Using Excel: {xl_path}")

    sheet = args.sheet if args.sheet is not None else pick_best_sheet(xl_path)
    print(f"Using sheet: {sheet}")

    df = pd.read_excel(xl_path, sheet_name=sheet, engine="openpyxl")
    df = normalize_columns(df).dropna(how="all")
    for c in CANONICAL_COLS:
        if c in df.columns:
            df[c] = robust_to_numeric(df[c])
    df = df.dropna(how="any")
    if len(df) < 6:
        print("Warning: dataset is tiny; scores will be noisy.")

    models, reports = train_ompt(df, verbose=True)

    bundle = {
        "models": models,
        "reports": [r.__dict__ for r in reports],
        "columns": CANONICAL_COLS,
        "excel_used": xl_path,
        "sheet_used": sheet,
        "n_rows": len(df),
    }
    with open(args.bundle, "wb") as f:
        pickle.dump(bundle, f)

    print("\n=== Cross-validated Scores (Leave-One-Out style) ===")
    print(pd.DataFrame([r.__dict__ for r in reports]).to_string(index=False))
    print(f"\nSaved bundle -> {args.bundle}")

if __name__ == "__main__":
    main()
