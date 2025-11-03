
"""
Train regression models for airline delay prediction (Task 3, D195).
- Reads CSV with columns like: year, month, carrier, airport, arr_delay, ...
- Trains LinearRegression and RandomForestRegressor inside a Pipeline.
- Saves the best pipeline (by R^2) to outputs/model/fd_model.pkl
- Writes metrics to outputs/model/metrics.json
- Saves a feature-importance chart if the model supports it.

Usage (from project root):
    python scripts/train_model.py --data "data/raw/Airline_Delay_Cause.csv"
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import matplotlib.pyplot as plt

def load_and_prepare(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    candidate_cols = ["arr_delay", "carrier", "airport", "month"]
    missing = [c for c in candidate_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns {missing}. Update mapping in load_and_prepare().")
    df = df[candidate_cols].copy().dropna()
    df["arr_delay"] = df["arr_delay"].clip(lower=-60, upper=360)
    df = df.rename(columns={"arr_delay": "DEP_DELAY",
                            "carrier": "OP_UNIQUE_CARRIER",
                            "airport": "ORIGIN",
                            "month": "MONTH"})
    return df

def build_pipelines():
    num_features = ["MONTH"]
    cat_features = ["OP_UNIQUE_CARRIER", "ORIGIN"]
    pre = ColumnTransformer([
        ("num", "passthrough", num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
    ])
    lin = Pipeline([("prep", pre), ("model", LinearRegression())])
    rfr = Pipeline([("prep", pre), ("model", RandomForestRegressor(n_estimators=300, random_state=42))])
    return {"LinearRegression": lin, "RandomForest": rfr}

def main(args):
    data_path = Path(args.data)
    out_dir = Path("outputs/model")
    charts_dir = Path("outputs/charts")
    out_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_prepare(str(data_path))
    y = df["DEP_DELAY"].values
    X = df[["MONTH", "OP_UNIQUE_CARRIER", "ORIGIN"]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = build_pipelines()
    results = {}
    best_name, best_r2 = None, -1e9

    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        r2 = float(r2_score(y_test, preds))
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))
        results[name] = {"R2": r2, "RMSE": rmse, "MAE": mae}
        if r2 > best_r2:
            best_r2 = r2
            best_name = name
            best_pipe = pipe

    model_path = out_dir / "fd_model.pkl"
    joblib.dump(best_pipe, model_path)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"results": results, "best_model": best_name}, f, indent=2)

    if best_name == "RandomForest":
        prep = best_pipe.named_steps["prep"]
        ohe = prep.named_transformers_["cat"]
        num_feats = ["MONTH"]
        cat_feats = list(ohe.get_feature_names_out(["OP_UNIQUE_CARRIER", "ORIGIN"]))
        feat_names = num_feats + cat_feats
        importances = best_pipe.named_steps["model"].feature_importances_
        idx = np.argsort(importances)[-20:]
        plt.figure(figsize=(8,6))
        plt.barh(np.array(feat_names)[idx], importances[idx])
        plt.title("Top Feature Importances (RandomForest)")
        plt.tight_layout()
        plt.savefig(charts_dir / "feature_importance.png", dpi=150)

    print("Saved model to:", model_path)
    print("Metrics:", json.dumps(results, indent=2))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/raw/Airline_Delay_Cause.csv")
    args = p.parse_args()
    main(args)
