
"""
Evaluate saved model on a fresh split and produce Task 3 visuals.

Usage:
    python scripts/evaluate_model.py --data "data/raw/Airline_Delay_Cause.csv"
Outputs:
    - outputs/charts/pred_vs_actual.png
    - prints R2 / RMSE / MAE
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

def load_and_prepare(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    needed = ["arr_delay", "carrier", "airport", "month"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing {missing}.")
    df = df[needed].dropna().copy()
    df['arr_delay'] = df['arr_delay'].clip(-60, 360)
    df = df.rename(columns={'arr_delay':'DEP_DELAY',
                            'carrier':'OP_UNIQUE_CARRIER',
                            'airport':'ORIGIN',
                            'month':'MONTH'})
    return df

def main(args):
    model_path = Path('outputs/model/fd_model.pkl')
    if not model_path.exists():
        raise FileNotFoundError('Model not found. Run scripts/train_model.py first.')
    pipe = joblib.load(model_path)
    charts_dir = Path('outputs/charts')
    charts_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_prepare(args.data)
    y = df['DEP_DELAY'].values
    X = df[['MONTH','OP_UNIQUE_CARRIER','ORIGIN']]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    preds = pipe.predict(Xte)
    r2 = float(r2_score(yte, preds))
    rmse = float(np.sqrt(mean_squared_error(yte, preds)))
    mae = float(mean_absolute_error(yte, preds))
    print({'R2': r2, 'RMSE': rmse, 'MAE': mae})

    plt.figure(figsize=(6,6))
    plt.scatter(yte, preds, s=10, alpha=0.6)
    lims = [min(min(yte), min(preds)), max(max(yte), max(preds))]
    plt.plot(lims, lims, '--')
    plt.xlabel('Actual Delay (min)')
    plt.ylabel('Predicted Delay (min)')
    plt.title('Predicted vs. Actual Delay')
    plt.tight_layout()
    plt.savefig(charts_dir/'pred_vs_actual.png', dpi=150)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/raw/Airline_Delay_Cause.csv')
    args = p.parse_args()
    main(args)
