"""
train_model.py
--------------
Loads the house price dataset, performs EDA, trains a Random Forest Regressor,
evaluates it, and saves the trained model + feature metadata to model.pkl.

Run once before starting the Flask API:
    python backend/train_model.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Paths ---
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "..", "data", "house_price_regression_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
PLOTS_DIR  = os.path.join(BASE_DIR, "..", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- 1. Load Data ---
print("=" * 60)
print("HOUSE PRICE PREDICTION -- MODEL TRAINING")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"\n[1] Dataset loaded -> {df.shape[0]:,} rows x {df.shape[1]} columns")
print(df.dtypes)
print("\nFirst 5 rows:")
print(df.head())

# --- 2. EDA ---
print("\n[2] Basic statistics:")
print(df.describe().round(2))

missing = df.isnull().sum()
print(f"\nMissing values:\n{missing[missing > 0] if missing.any() else 'None'}")

# Correlation heatmap
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
ax.set_title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"), dpi=120)
plt.close()
print("  -> Saved correlation_heatmap.png")

# Target distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df["House_Price"], bins=40, color="#3b82d4", edgecolor="white")
axes[0].set_title("House Price Distribution")
axes[0].set_xlabel("Price ($)")
axes[1].hist(np.log1p(df["House_Price"]), bins=40, color="#7c5cd8", edgecolor="white")
axes[1].set_title("Log-transformed Price Distribution")
axes[1].set_xlabel("log(Price + 1)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "price_distribution.png"), dpi=120)
plt.close()
print("  -> Saved price_distribution.png")

# --- 3. Prepare Features ---
FEATURES = ["Square_Footage", "Num_Bedrooms", "Num_Bathrooms",
            "Year_Built", "Lot_Size", "Garage_Size", "Neighborhood_Quality"]
TARGET   = "House_Price"

X = df[FEATURES].values
y = df[TARGET].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"\n[3] Train/test split -> {len(X_train):,} train / {len(X_test):,} test")

# --- 4. Train & Compare Models ---
print("\n[4] Training models ...")

candidates = {
    "Linear Regression":  LinearRegression(),
    "Gradient Boosting":  GradientBoostingRegressor(n_estimators=200, random_state=42),
    "Random Forest":      RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
}

results = {}
for name, mdl in candidates.items():
    mdl.fit(X_train, y_train)
    preds = mdl.predict(X_test)
    results[name] = {
        "MAE":  mean_absolute_error(y_test, preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
        "R2":   r2_score(y_test, preds),
    }
    print(f"  {name:25s}  MAE={results[name]['MAE']:>10,.0f}  "
          f"RMSE={results[name]['RMSE']:>10,.0f}  R2={results[name]['R2']:.4f}")

# Pick best by R2
best_name  = max(results, key=lambda k: results[k]["R2"])
best_model = candidates[best_name]
print(f"\n  Best model: {best_name}  (R2 = {results[best_name]['R2']:.4f})")

# Feature importances (tree models only)
if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
    fig, ax = plt.subplots(figsize=(8, 5))
    sorted_idx = np.argsort(importances)
    ax.barh([FEATURES[i] for i in sorted_idx], importances[sorted_idx], color="#3b82d4")
    ax.set_title(f"Feature Importances - {best_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "feature_importances.png"), dpi=120)
    plt.close()
    print("  -> Saved feature_importances.png")

# --- 5. Save Artefact ---
artefact = {
    "model":      best_model,
    "features":   FEATURES,
    "metrics":    results[best_name],
    "model_name": best_name,
}
with open(MODEL_PATH, "wb") as f:
    pickle.dump(artefact, f)

print(f"\n[5] Model saved -> {MODEL_PATH}")
print("=" * 60)
print("Training complete.  Start the API with:  python backend/app.py")
print("=" * 60)
