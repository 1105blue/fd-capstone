import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

import matplotlib.pyplot as plt

# === Path to your dataset ===
DATA_PATH = r"C:\fd-capstone\data\raw\On_Time_Reporting_2023_01.csv\Airline_Delay_Cause.csv"

# === Load & preview ===
df = pd.read_csv(DATA_PATH, low_memory=False)
print("Shape:", df.shape)
print("Columns:", list(df.columns[:25]))

# === Select and rename columns to standard names ===
# Mapping your dataset's fields to expected model features
cols = ["arr_delay", "carrier", "airport", "month"]
df_small = df[cols].copy()
df_small.columns = ["DEP_DELAY", "OP_UNIQUE_CARRIER", "ORIGIN", "MONTH"]

# === Basic cleaning ===
df_small = df_small.dropna(subset=["DEP_DELAY", "OP_UNIQUE_CARRIER", "ORIGIN", "MONTH"])
# Keep only reasonable delays between -60 and 360 minutes
df_small = df_small[df_small["DEP_DELAY"].between(-60, 360)]
print("Cleaned data shape:", df_small.shape)

# === Define model variables ===
y = df_small["DEP_DELAY"]
X = df_small[["MONTH", "OP_UNIQUE_CARRIER", "ORIGIN"]]

num_features = ["MONTH"]
cat_features = ["OP_UNIQUE_CARRIER", "ORIGIN"]

pre = ColumnTransformer([
    ("num", "passthrough", num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
])

model = Pipeline([
    ("prep", pre),
    ("linreg", LinearRegression()),
])

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)

# === Metrics ===
r2 = r2_score(y_test, preds)
mse = mean_squared_error(y_test, preds)
print(f"\nModel Evaluation:")
print(f"R² = {r2:.4f}")
print(f"MSE = {mse:.4f}")

# === Charts (for Task 3 visuals) ===
Path("outputs").mkdir(exist_ok=True)

plt.figure()
df_small["DEP_DELAY"].plot(kind="hist", bins=60, title="Delay Distribution (Minutes)")
plt.xlabel("Minutes")
plt.tight_layout()
plt.savefig("outputs/delay_distribution.png", dpi=150)
plt.show()

plt.figure()
df_small.groupby("MONTH")["DEP_DELAY"].mean().plot(kind="bar", title="Average Delay by Month")
plt.ylabel("Minutes")
plt.tight_layout()
plt.savefig("outputs/avg_delay_by_month.png", dpi=150)
plt.show()

print("\n✅ Charts saved to: outputs/delay_distribution.png and outputs/avg_delay_by_month.png")
