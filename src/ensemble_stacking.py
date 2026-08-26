"""
Ensemble Stacking Model
Blends Linear Regression, Random Forest, and XGBoost using a Ridge meta-learner.
"""
import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from data_preprocessing import load_data
from feature_engineering import build_preprocessor

print("📥 Loading data...")
import os
data_path = os.path.join(os.path.dirname(__file__), "..", "data", "vehicles_cleaned.csv")
df = load_data(data_path)
df = df.sample(frac=0.3, random_state=42)
print(f"   Rows used: {len(df)}")

X = df.drop(["price", "log_price"], axis=1)
y = df["log_price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load pre-trained models
models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
linear_pipe = joblib.load(os.path.join(models_dir, "linear_log.pkl"))
rf_pipe = joblib.load(os.path.join(models_dir, "random_forest_log.pkl"))
xgb_pipe = joblib.load(os.path.join(models_dir, "xgboost_log.pkl"))

print("🔀 Generating base model predictions for meta-learner...")

# Get out-of-fold style predictions on test set for meta-learner training
linear_preds = linear_pipe.predict(X_train)
rf_preds = rf_pipe.predict(X_train)
xgb_preds = xgb_pipe.predict(X_train)

# Stack predictions as features for meta-learner
meta_features_train = np.column_stack([linear_preds, rf_preds, xgb_preds])

# Also get test set meta-features
linear_preds_test = linear_pipe.predict(X_test)
rf_preds_test = rf_pipe.predict(X_test)
xgb_preds_test = xgb_pipe.predict(X_test)

meta_features_test = np.column_stack([linear_preds_test, rf_preds_test, xgb_preds_test])

# Train Ridge meta-learner
print("🧠 Training Ridge meta-learner...")
meta_learner = Ridge(alpha=1.0)
meta_learner.fit(meta_features_train, y_train)

# Evaluate
from sklearn.metrics import mean_absolute_error, r2_score

# Individual model scores
for name, preds in [("Linear", linear_preds_test), ("Random Forest", rf_preds_test), ("XGBoost", xgb_preds_test)]:
    actual = np.expm1(y_test)
    pred_prices = np.expm1(preds)
    mae = mean_absolute_error(actual, pred_prices)
    r2 = r2_score(actual, pred_prices)
    print(f"   {name}: MAE=${mae:,.0f}, R²={r2:.4f}")

# Ensemble score
ensemble_preds = meta_learner.predict(meta_features_test)
ensemble_actual = np.expm1(y_test)
ensemble_prices = np.expm1(ensemble_preds)
ensemble_mae = mean_absolute_error(ensemble_actual, ensemble_prices)
ensemble_r2 = r2_score(ensemble_actual, ensemble_prices)
print(f"   🏆 Ensemble: MAE=${ensemble_mae:,.0f}, R²={ensemble_r2:.4f}")

# Save the full ensemble pipeline
ensemble_data = {
    "linear": linear_pipe,
    "random_forest": rf_pipe,
    "xgboost": xgb_pipe,
    "meta_learner": meta_learner,
}

joblib.dump(ensemble_data, os.path.join(models_dir, "ensemble_stacking.pkl"))
print("\n✅ Saved ensemble_stacking.pkl")
