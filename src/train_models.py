import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from data_preprocessing import load_data
from feature_engineering import build_preprocessor

print("📥 Loading data...")
df = load_data("../data/vehicles_cleaned.csv")

df = df.sample(frac=0.3, random_state=42)
print(f" Rows used: {len(df)}")

X = df.drop(["price", "log_price"], axis=1)
y = df["log_price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

preprocessor = build_preprocessor()

models = {
    "linear": LinearRegression(),

    "random_forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=16,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42
    ),

    "xgboost": XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42
    )
}

for name, model in models.items():
    print(f"\n Training {name}")

    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)
    joblib.dump(pipe, f"../models/{name}_log.pkl")
    print(f" Saved {name}_log.pkl")

print("\n ALL MODELS TRAINED")
