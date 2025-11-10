#!/usr/bin/env python3
import pickle
import re
from difflib import get_close_matches
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

# 🔧 HARD-CODED FILE PATH (edit this line only)
EXCEL_FILE = r"C:\ML_PRED\Book3.xlsx"     # <-- change to your actual Excel path
SHEET_NAME = "Sheet1"
OUTPUT_BUNDLE = "model_bundle.pkl"

# Canonical column names expected
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

def clean_header(h: str) -> str:
    return re.sub(r"\s+", " ", h.strip()).replace("’", "'")

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    original = list(df.columns)
    rename_map: Dict[str, str] = {}
    for col in df.columns:
        key = clean_header(col).lower()
        mapped = None
        # Regex matching
        for patt, canon in ALIASES.items():
            if re.fullmatch(patt, key):
                mapped = canon
                break
        # Fuzzy matching fallback
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
        df[c] = (df[c].astype(str)
                       .str.replace(",", "", regex=False)
                       .str.replace(" ", "", regex=False))
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def build_candidates():
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler())])
    pre = ColumnTransformer([("num", numeric, list(range(0, 10)))], remainder="drop")
    hgb = HistGradientBoostingRegressor(random_state=42, learning_rate=0.07, max_leaf_nodes=31, min_samples_leaf=2)
    rf = RandomForestRegressor(n_estimators=600, random_state=42, n_jobs=-1)
    return {
        "HistGradientBoosting": Pipeline([("pre", pre), ("reg", hgb)]),
        "RandomForest": Pipeline([("pre", pre), ("reg", rf)]),
    }

def loocv(model, X, y):
    n = len(X)
    kf = KFold(n_splits=n, shuffle=True, random_state=42)
    r2s, maes = [], []
    for tr, te in kf.split(X):
        model.fit(X.iloc[tr], y.iloc[tr])
        pred = model.predict(X.iloc[te])
        r2s.append(r2_score(y.iloc[te], pred))
        maes.append(mean_absolute_error(y.iloc[te], pred))
    return float(np.mean(r2s)), float(np.std(r2s)), float(np.mean(maes)), float(np.std(maes))

def main():
    print(f"📂 Using Excel: {EXCEL_FILE}")
    print(f"📑 Using sheet: {SHEET_NAME}")
    df = pd.read_excel(EXCEL_FILE, sheet_name=SHEET_NAME, engine="openpyxl")

    df = normalize_columns(df)
    df = numericify(df)
    df = df.dropna(thresh=2)
    if len(df) < 3:
        print("⚠️ Warning: dataset is tiny; scores may be unstable.\n")

    models = {}
    reports = []
    for target in CANONICAL_COLS:
        if target not in df.columns:
            continue
        X = df.drop(columns=[target])
        y = df[target]
        mask = y.notna()
        X, y = X[mask], y[mask]
        if len(X) < 3:
            print(f"[{target}] ⏭️ skipped (too few rows).")
            continue

        best = None
        best_name = ""
        for name, cand in build_candidates().items():
            r2m, r2s, maem, maes = loocv(cand, X, y)
            print(f"[{target}] {name} → R2={r2m:.3f}±{r2s:.3f}  MAE={maem:.3f}±{maes:.3f}")
            if (best is None) or (r2m > best["r2"]) or (abs(r2m - best["r2"]) < 1e-9 and maem < best["mae"]):
                best = {"model": cand, "r2": r2m, "mae": maem, "r2s": r2s, "maes": maes}
                best_name = name

        if best:
            best["model"].fit(X, y)
            models[target] = best["model"]
            reports.append({
                "target": target,
                "model_name": best_name,
                "r2_mean": best["r2"],
                "mae_mean": best["mae"],
                "r2_std": best["r2s"],
                "mae_std": best["maes"]
            })

    bundle = {"models": models, "reports": reports, "columns": CANONICAL_COLS}
    with open(OUTPUT_BUNDLE, "wb") as f:
        pickle.dump(bundle, f)

    print("\n=== ✅ Cross-validated Scores (LOOCV) ===")
    if reports:
        print(pd.DataFrame(reports).to_string(index=False))
    else:
        print("No models were trained — please check the Excel data.")
    print(f"\n💾 Saved trained bundle → {OUTPUT_BUNDLE}")

if __name__ == "__main__":
    main()
