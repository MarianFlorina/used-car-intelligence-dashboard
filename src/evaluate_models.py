import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from data_preprocessing import load_data

df = load_data("../data/vehicles_cleaned.csv")

X = df.drop(["price", "log_price"], axis=1)
y_log = df["log_price"]
y_actual = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)

y_test_actual = y_actual.loc[y_test.index]

models = {
    "Linear Regression": "../models/linear_log.pkl",
    "Random Forest": "../models/random_forest_log.pkl",
    "XGBoost": "../models/xgboost_log.pkl"
}

results = []

for name, path in models.items():
    model = joblib.load(path)
    preds = np.expm1(model.predict(X_test))

    results.append([
        name,
        mean_absolute_error(y_test_actual, preds),
        np.sqrt(mean_squared_error(y_test_actual, preds)),
        r2_score(y_test_actual, preds)
    ])

df_results = pd.DataFrame(results, columns=["Model", "MAE", "RMSE", "R2"])
print("\n📊 FINAL MODEL COMPARISON")
print(df_results)
