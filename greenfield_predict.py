#!/usr/bin/env python3
import argparse
import json
import pickle
from typing import Dict

import numpy as np
import pandas as pd

CANONICAL_COLS = [
    "Client Revenue",
    "Number of Users",
    "RICEFW",
    "Duration (Months)",
    "Countries/Market",
    "Estimated Effort (man days)",
]

def load_bundle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def predict_missing(bundle_path: str, known_json: str) -> Dict[str, float]:
    bundle = load_bundle(bundle_path)
    models = bundle["models"]
    known: Dict[str, float] = json.loads(known_json)

    vals = {c: np.nan for c in CANONICAL_COLS}
    for k, v in known.items():
        if k in vals and v is not None:
            vals[k] = float(v)

    for _ in range(50):
        changed = False
        for target, model in models.items():
            x_cols = [c for c in CANONICAL_COLS if c != target]
            x = pd.DataFrame([[vals[c] if not np.isnan(vals[c]) else 0.0 for c in x_cols]], columns=x_cols)
            pred = float(model.predict(x)[0])
            if (np.isnan(vals[target])) and target not in known:
                vals[target] = pred
                changed = True
        if not changed:
            break
    return vals

def main():
    ap = argparse.ArgumentParser(description="Predict any missing fields from partial inputs using the trained bundle.")
    ap.add_argument("--bundle", default="model_bundle.pkl", help="Path to model bundle (defaults to ./model_bundle.pkl).")
    ap.add_argument("--inputs", required=True, help='JSON string, e.g. {"Client Revenue": 34, "Number of Users": 7500}')
    args = ap.parse_args()

    out = predict_missing(args.bundle, args.inputs)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
