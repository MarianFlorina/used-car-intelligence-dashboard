from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

app = Flask(__name__)



MODEL_PATH = "../models/random_forest_log.pkl"
model = joblib.load(MODEL_PATH)



cars_df = pd.read_csv(
    "../data/vehicles_cleaned.csv",
    engine="python",
    on_bad_lines="skip"
)

cars_df["manufacturer"] = cars_df["manufacturer"].astype(str).str.lower()

cars_df = cars_df[cars_df["manufacturer"] != "unknown"]



CURRENCY_RATES = {
    "USD": 1.0,
    "INR": 83.0,
    "EUR": 0.92,
    "GBP": 0.78
}

MODEL_MAE_USD = 3400



@app.route("/")
def index():
    return render_template("index.html")



@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        # ---------- INPUTS ----------
        year = int(data["year"])
        odometer = int(data["odometer"])
        manufacturer = data["manufacturer"].strip().lower()
        fuel = data["fuel"]
        transmission = data["transmission"]
        drive = data["drive"]
        vehicle_type = data["type"]
        condition = data["condition"]
        currency = data["currency"]

        listed_price = data.get("listed_price")
        listed_price = float(listed_price) if listed_price else None

        current_year = datetime.now().year
        vehicle_age = current_year - year



        if year < 1980 or year > current_year:
            return jsonify({"error": "Invalid manufacturing year"}), 400

        if odometer < 0 or odometer > 1_000_000:
            return jsonify({"error": "Mileage must be between 0 and 1,000,000 miles"}), 400

        if not manufacturer.isalpha():
            return jsonify({"error": "Manufacturer must contain only alphabets"}), 400



        input_df = pd.DataFrame([{
            "vehicle_age": vehicle_age,
            "age_squared": vehicle_age ** 2,
            "log_odometer": np.log1p(odometer),
            "manufacturer": manufacturer,
            "fuel": fuel,
            "transmission": transmission,
            "drive": drive,
            "type": vehicle_type,
            "condition": condition
        }])



        log_price = model.predict(input_df)[0]
        price_usd = np.expm1(log_price)

        predicted_price = round(price_usd * CURRENCY_RATES[currency], 2)



        lower = max(0, price_usd - MODEL_MAE_USD)
        upper = price_usd + MODEL_MAE_USD

        price_range = [
            round(lower * CURRENCY_RATES[currency], 2),
            round(upper * CURRENCY_RATES[currency], 2)
        ]



        confidence = max(60, min(95, 100 - (MODEL_MAE_USD / price_usd * 100)))
        confidence = round(confidence, 1)



        deal_status = "N/A"
        recommendation = "N/A"

        if listed_price:

            predicted_currency = price_usd * CURRENCY_RATES[currency]

            diff_pct = ((listed_price - predicted_currency) / predicted_currency) * 100

            if diff_pct <= -10:
                deal_status = "Good Deal"
                recommendation = "Buy"

            elif diff_pct <= 10:
                deal_status = "Fair Deal"
                recommendation = "Consider"

            else:
                deal_status = "Overpriced"
                recommendation = "Avoid"



        similar_cars = cars_df[
            (cars_df["price"].between(price_usd * 0.9, price_usd * 1.1))
        ]



        recommended_manufacturers = (
            similar_cars["manufacturer"]
            .value_counts()
            .head(5)
            .index
            .str.title()
            .tolist()
        )



        similar_cars_table = similar_cars[
            ["manufacturer", "year", "odometer", "price"]
        ].head(5)

        similar_cars_list = similar_cars_table.to_dict(orient="records")



        return jsonify({

            "predicted_price": predicted_price,
            "currency": currency,
            "price_range": price_range,
            "vehicle_age": vehicle_age,
            "deal_status": deal_status,
            "recommendation": recommendation,
            "confidence": confidence,
            "recommended_manufacturers": recommended_manufacturers,
            "similar_cars": similar_cars_list

        })


    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(debug=True)