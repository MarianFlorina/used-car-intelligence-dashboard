import joblib
import numpy as np
import matplotlib.pyplot as plt
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

model = joblib.load("../models/random_forest_log.pkl")
preds = np.expm1(model.predict(X_test))

plt.figure(figsize=(7,7))
plt.scatter(y_test_actual, preds, alpha=0.3)
plt.plot([y_test_actual.min(), y_test_actual.max()],
         [y_test_actual.min(), y_test_actual.max()])
plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.title("Actual vs Predicted Prices (Random Forest)")
plt.tight_layout()
plt.show()
