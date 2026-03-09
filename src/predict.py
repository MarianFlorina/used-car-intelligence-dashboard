import joblib
import numpy as np
import pandas as pd

# Load model only once (faster)
MODEL_PATH = "../models/random_forest_log.pkl"
model = joblib.load(MODEL_PATH)


def predict_price(
    year,
    odometer,
    manufacturer,
    fuel,
    transmission,
    drive,
    vehicle_type,
    condition
):
    """
    Predict used car price in USD
    """

    # Feature engineering
    vehicle_age = 2025 - year
    age_squared = vehicle_age ** 2
    log_odometer = np.log1p(odometer)

    # Create DataFrame for model input
    input_df = pd.DataFrame([{
        "vehicle_age": vehicle_age,
        "age_squared": age_squared,
        "log_odometer": log_odometer,
        "manufacturer": manufacturer,
        "fuel": fuel,
        "transmission": transmission,
        "drive": drive,
        "type": vehicle_type,
        "condition": condition
    }])

    # Predict log(price)
    log_price_pred = model.predict(input_df)[0]

    # Convert log price → actual price
    price_pred = np.expm1(log_price_pred)

    return round(price_pred, 2)


# CLI Test Mode
if __name__ == "__main__":
    price = predict_price(
        year=2016,
        odometer=90000,
        manufacturer="ford",
        fuel="diesel",
        transmission="manual",
        drive="4wd",
        vehicle_type="truck",
        condition="fair"
    )

    print(f"Estimated Used Car Price: ${price}")