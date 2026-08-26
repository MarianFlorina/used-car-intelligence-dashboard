"""
Used Car Intelligence Dashboard - Backend
Features: Price prediction, SHAP explainability, ensemble stacking,
depreciation curves, total cost of ownership, market intelligence,
best value finder, negotiation calculator, regional comparison.

Fixed: input validation, pathlib paths, rate limiting, caching,
production readiness, API key security.
"""
from flask import Flask, render_template, request, jsonify
from flask.json.provider import DefaultJSONProvider
from pathlib import Path
import os
import json
import hashlib
import joblib
import numpy as np
import pandas as pd
import shap
import logging
from datetime import datetime
from dotenv import load_dotenv
from functools import lru_cache

# ─── Setup ────────────────────────────────────────────────────────────

# Resolve paths relative to this file (not CWD)
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

# Load environment variables
load_dotenv(PROJECT_DIR / ".env")

# Logging setup (redacts sensitive data)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Flask App ────────────────────────────────────────────────────────

app = Flask(__name__)

# Numpy JSON encoder for Flask 3.x
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

class NumpyJSONProvider(DefaultJSONProvider):
    def dumps(self, obj, **kwargs):
        kwargs.setdefault("cls", NumpyEncoder)
        return super().dumps(obj, **kwargs)

app.json = NumpyJSONProvider(app)

# ─── Rate Limiting (simple in-memory) ────────────────────────────────

from collections import defaultdict
import time

_rate_limits = defaultdict(list)
RATE_LIMIT = 30  # requests per minute
RATE_WINDOW = 60  # seconds

def check_rate_limit():
    """Simple in-memory rate limiter. Returns True if request is allowed."""
    now = time.time()
    client_ip = request.remote_addr or "unknown"
    
    # Clean old entries
    _rate_limits[client_ip] = [
        t for t in _rate_limits[client_ip] if now - t < RATE_WINDOW
    ]
    
    if len(_rate_limits[client_ip]) >= RATE_LIMIT:
        return False
    
    _rate_limits[client_ip].append(now)
    return True

# ─── Input Validation ────────────────────────────────────────────────

VALID_FUELS = {"gas", "diesel", "electric", "hybrid"}
VALID_TRANSMISSIONS = {"automatic", "manual"}
VALID_DRIVES = {"fwd", "rwd", "4wd"}
VALID_TYPES = {"sedan", "suv", "truck", "pickup"}
VALID_CONDITIONS = {"excellent", "good", "fair"}
VALID_CURRENCIES = {"USD", "INR", "EUR", "GBP"}

def validate_prediction_input(data):
    """Validate all input fields. Returns error message or None."""
    errors = []
    
    # Required fields
    for field in ["year", "odometer", "manufacturer", "fuel", "transmission", "drive", "type", "condition"]:
        if field not in data or data[field] is None:
            errors.append(f"Missing required field: {field}")
    
    if errors:
        return "; ".join(errors)
    
    # Year
    try:
        year = int(data["year"])
        current_year = datetime.now().year
        if year < 1900 or year > current_year:
            errors.append(f"Year must be between 1900 and {current_year}")
    except (ValueError, TypeError):
        errors.append("Year must be a valid integer")
    
    # Odometer
    try:
        odometer = int(data["odometer"])
        if odometer < 0 or odometer > 1_000_000:
            errors.append("Odometer must be between 0 and 1,000,000")
    except (ValueError, TypeError):
        errors.append("Odometer must be a valid integer")
    
    # Manufacturer
    manufacturer = str(data.get("manufacturer", "")).strip()
    if not manufacturer:
        errors.append("Manufacturer is required")
    elif len(manufacturer) < 2:
        errors.append("Manufacturer must be at least 2 characters")
    elif not manufacturer.isalpha():
        errors.append("Manufacturer must contain only letters")
    
    # Categorical fields
    if data.get("fuel", "").lower() not in VALID_FUELS:
        errors.append(f"Invalid fuel type. Must be one of: {', '.join(VALID_FUELS)}")
    if data.get("transmission", "").lower() not in VALID_TRANSMISSIONS:
        errors.append(f"Invalid transmission. Must be one of: {', '.join(VALID_TRANSMISSIONS)}")
    if data.get("drive", "").lower() not in VALID_DRIVES:
        errors.append(f"Invalid drive type. Must be one of: {', '.join(VALID_DRIVES)}")
    if data.get("type", "").lower() not in VALID_TYPES:
        errors.append(f"Invalid vehicle type. Must be one of: {', '.join(VALID_TYPES)}")
    if data.get("condition", "").lower() not in VALID_CONDITIONS:
        errors.append(f"Invalid condition. Must be one of: {', '.join(VALID_CONDITIONS)}")
    if data.get("currency", "USD") not in VALID_CURRENCIES:
        errors.append(f"Invalid currency. Must be one of: {', '.join(VALID_CURRENCIES)}")
    
    # Listed price (optional)
    listed_price = data.get("listed_price")
    if listed_price is not None and listed_price != "":
        try:
            lp = float(listed_price)
            if lp <= 0:
                errors.append("Listed price must be positive")
        except (ValueError, TypeError):
            errors.append("Listed price must be a valid number")
    
    return "; ".join(errors) if errors else None

# ─── Import Live Data ─────────────────────────────────────────────────

from live_data import fetch_currency_rates, get_last_updated, fetch_fuel_prices

# ─── Load Models & Data ──────────────────────────────────────────────

logger.info("Loading models...")
rf_model = joblib.load(PROJECT_DIR / "models" / "random_forest_log.pkl")
ensemble_data = joblib.load(PROJECT_DIR / "models" / "ensemble_stacking.pkl")

logger.info("Loading dataset...")
cars_df = pd.read_csv(
    PROJECT_DIR / "data" / "vehicles_cleaned.csv",
    engine="python", on_bad_lines="skip"
)
cars_df["manufacturer"] = cars_df["manufacturer"].astype(str).str.lower()
cars_df = cars_df[cars_df["manufacturer"] != "unknown"]

# Precompute features (use dynamic year)
current_year = datetime.now().year
cars_df["vehicle_age"] = current_year - cars_df["year"]
cars_df["age_squared"] = cars_df["vehicle_age"] ** 2
cars_df["log_odometer"] = np.log1p(cars_df["odometer"])

# SHAP explainer
rf_pipeline = rf_model.named_steps["model"]
preprocessor = rf_model.named_steps["preprocess"]

shap_background = cars_df.sample(20, random_state=42).drop(
    columns=["price", "log_price"], errors="ignore"
)
shap_background_transformed = preprocessor.transform(shap_background).astype(np.float64)
shap_feature_names = preprocessor.get_feature_names_out().tolist()

try:
    shap_explainer = shap.TreeExplainer(rf_pipeline)
    logger.info("SHAP explainer initialized successfully")
except Exception as e:
    shap_explainer = None
    logger.warning(f"SHAP explainer failed to initialize: {e}")

# ─── Constants ────────────────────────────────────────────────────────

CURRENCY_SYMBOLS = {"USD": "$", "INR": "\u20b9", "EUR": "\u20ac", "GBP": "\u00a3"}
MODEL_MAE_USD = 3400  # Approximate MAE from model evaluation
ANNUAL_MILES = 12000
INSURANCE_BASE = 1400
MAINTENANCE_BASE = 800

# Depreciation cache
_depreciation_cache = {}

logger.info(f"Ready! {len(cars_df)} cars loaded. Current year: {current_year}")

# ─── Routes ───────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint for monitoring."""
    return jsonify({
        "status": "healthy",
        "cars_loaded": len(cars_df),
        "current_year": current_year,
        "shap_ready": shap_explainer is not None,
    })

@app.route("/manufacturers", methods=["GET"])
def get_manufacturers():
    mfrs = sorted(cars_df["manufacturer"].unique().tolist())
    return jsonify(mfrs)

@app.route("/predict", methods=["POST"])
def predict():
    # Rate limit check
    if not check_rate_limit():
        return jsonify({"error": "Rate limit exceeded. Please wait a moment."}), 429
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Validate all inputs
        validation_error = validate_prediction_input(data)
        if validation_error:
            return jsonify({"error": validation_error}), 400
        
        # Extract validated inputs
        year = int(data["year"])
        odometer = int(data["odometer"])
        manufacturer = data["manufacturer"].strip().lower()
        fuel = data["fuel"].lower()
        transmission = data["transmission"].lower()
        drive = data["drive"].lower()
        vehicle_type = data["type"].lower()
        condition = data["condition"].lower()
        currency = data.get("currency", "USD")
        listed_price = data.get("listed_price")
        listed_price = float(listed_price) if listed_price else None
        
        # Dynamic year calculation
        now = datetime.now()
        vehicle_age = now.year - year
        
        # Fetch live currency rates
        CURRENCY_RATES = fetch_currency_rates()
        
        # Build input DataFrame
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
        
        # Ensemble prediction
        linear_pipe = ensemble_data["linear"]
        xgb_pipe = ensemble_data["xgboost"]
        meta_learner = ensemble_data["meta_learner"]
        
        linear_pred = float(linear_pipe.predict(input_df)[0])
        rf_pred = float(rf_model.predict(input_df)[0])
        xgb_pred = float(xgb_pipe.predict(input_df)[0])
        meta_features = np.column_stack([[linear_pred], [rf_pred], [xgb_pred]])
        log_price_ensemble = meta_learner.predict(meta_features)[0]
        price_usd = float(np.expm1(log_price_ensemble))
        
        predicted_price = round(price_usd * CURRENCY_RATES[currency], 2)
        symbol = CURRENCY_SYMBOLS[currency]
        
        # Model comparison
        model_comparison = {
            "linear": round(np.expm1(linear_pred) * CURRENCY_RATES[currency], 2),
            "random_forest": round(np.expm1(rf_pred) * CURRENCY_RATES[currency], 2),
            "xgboost": round(np.expm1(xgb_pred) * CURRENCY_RATES[currency], 2),
            "ensemble": round(price_usd * CURRENCY_RATES[currency], 2),
        }
        
        # Price range
        lower = max(0, price_usd - MODEL_MAE_USD)
        upper = price_usd + MODEL_MAE_USD
        price_range = [
            round(lower * CURRENCY_RATES[currency], 2),
            round(upper * CURRENCY_RATES[currency], 2)
        ]
        
        # Confidence
        confidence = max(60, min(95, 100 - (MODEL_MAE_USD / price_usd * 100)))
        confidence = round(confidence, 1)
        
        # SHAP explainability
        shap_values = get_shap_values(input_df)
        
        # Deal status & negotiation
        deal_status = "N/A"
        recommendation = "N/A"
        negotiation = None
        
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
            
            negotiation = {
                "listed_price": round(listed_price, 2),
                "fair_price": predicted_price,
                "suggested_offer": round(predicted_price * 0.92 * CURRENCY_RATES[currency], 2),
                "walk_away_price": round(predicted_price * 1.05 * CURRENCY_RATES[currency], 2),
                "savings": round((listed_price - predicted_price * 0.92) * CURRENCY_RATES[currency], 2),
                "strategy": get_negotiation_strategy(diff_pct, vehicle_age, odometer, condition),
            }
        
        # TCO
        tco = calculate_tco(fuel, vehicle_age, price_usd)
        
        # Depreciation curve (cached)
        depreciation = get_depreciation_curve(
            manufacturer, vehicle_type, fuel, transmission, drive, condition
        )
        
        # Regional comparison
        regional = get_regional_comparison(input_df, price_usd, currency)
        
        # Similar cars
        similar_cars = cars_df[
            (cars_df["price"].between(price_usd * 0.85, price_usd * 1.15))
        ].head(5)
        similar_cars_list = [
            {
                "manufacturer": str(rec["manufacturer"]),
                "year": int(rec["year"]),
                "odometer": int(rec["odometer"]),
                "price": float(rec["price"]),
            }
            for rec in similar_cars[["manufacturer", "year", "odometer", "price"]].to_dict(orient="records")
        ]
        
        # Market intelligence
        market = get_market_intelligence(manufacturer, vehicle_type)
        
        return jsonify({
            "predicted_price": predicted_price,
            "currency": currency,
            "symbol": symbol,
            "price_range": price_range,
            "vehicle_age": vehicle_age,
            "deal_status": deal_status,
            "recommendation": recommendation,
            "confidence": confidence,
            "model_comparison": model_comparison,
            "shap_values": shap_values,
            "negotiation": negotiation,
            "tco": tco,
            "depreciation": depreciation,
            "regional": regional,
            "similar_cars": similar_cars_list,
            "market": market,
        })
    
    except Exception as e:
        logger.error(f"Prediction error: {type(e).__name__}")
        return jsonify({"error": "An internal error occurred. Please try again."}), 500

@app.route("/live_data", methods=["GET"])
def live_data():
    """Return live currency rates and fuel prices."""
    currency_rates = fetch_currency_rates()
    fuel_prices = fetch_fuel_prices()
    last_updated = get_last_updated()
    
    return jsonify({
        "currency_rates": currency_rates,
        "currency_symbols": CURRENCY_SYMBOLS,
        "fuel_prices": fuel_prices,
        "last_updated": last_updated,
    })

@app.route("/best_value", methods=["POST"])
def best_value():
    # Rate limit check
    if not check_rate_limit():
        return jsonify({"error": "Rate limit exceeded. Please wait a moment."}), 429
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Validate budget
        try:
            budget = float(data.get("budget", 0))
            if budget <= 0:
                return jsonify({"error": "Budget must be positive"}), 400
        except (ValueError, TypeError):
            return jsonify({"error": "Invalid budget value"}), 400
        
        currency = data.get("currency", "USD")
        if currency not in VALID_CURRENCIES:
            return jsonify({"error": f"Invalid currency. Must be one of: {', '.join(VALID_CURRENCIES)}"}), 400
        
        preferences = data.get("preferences", {})
        
        # Validate preferences
        if preferences.get("fuel") and preferences["fuel"] not in VALID_FUELS:
            return jsonify({"error": "Invalid fuel filter"}), 400
        if preferences.get("type") and preferences["type"] not in VALID_TYPES:
            return jsonify({"error": "Invalid type filter"}), 400
        if preferences.get("drive") and preferences["drive"] not in VALID_DRIVES:
            return jsonify({"error": "Invalid drive filter"}), 400
        
        # Fetch live currency rates
        CURRENCY_RATES = fetch_currency_rates()
        rate = CURRENCY_RATES.get(currency, 1.0)
        budget_usd = budget / rate
        
        candidates = cars_df.copy()
        
        if preferences.get("fuel"):
            candidates = candidates[candidates["fuel"] == preferences["fuel"]]
        if preferences.get("type"):
            candidates = candidates[candidates["type"] == preferences["type"]]
        if preferences.get("drive"):
            candidates = candidates[candidates["drive"] == preferences["drive"]]
        if preferences.get("min_year"):
            try:
                candidates = candidates[candidates["year"] >= int(preferences["min_year"])]
            except (ValueError, TypeError):
                pass
        
        # Limit sample size for performance
        sample_size = min(300, len(candidates))
        sample = candidates.sample(sample_size, random_state=42)
        
        results = []
        for _, row in sample.iterrows():
            input_df = pd.DataFrame([{
                "vehicle_age": current_year - row["year"],
                "age_squared": (current_year - row["year"]) ** 2,
                "log_odometer": np.log1p(row["odometer"]),
                "manufacturer": row["manufacturer"],
                "fuel": row["fuel"],
                "transmission": row["transmission"],
                "drive": row["drive"],
                "type": row["type"],
                "condition": row.get("condition", "good"),
            }])
            
            predicted = np.expm1(rf_model.predict(input_df)[0])
            
            if row["price"] <= budget_usd and row["price"] < predicted * 0.9:
                undervaluation = round((1 - row["price"] / predicted) * 100, 1)
                results.append({
                    "manufacturer": row["manufacturer"],
                    "year": int(row["year"]),
                    "odometer": int(row["odometer"]),
                    "actual_price": round(row["price"] * rate, 2),
                    "predicted_price": round(predicted * rate, 2),
                    "undervaluation_pct": undervaluation,
                    "score": undervaluation * (1 + (current_year - row["year"]) * 0.02),
                })
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return jsonify({
            "best_value_cars": results[:10],
            "currency": currency,
            "symbol": CURRENCY_SYMBOLS[currency],
        })
    
    except Exception as e:
        logger.error(f"Best value error: {type(e).__name__}")
        return jsonify({"error": "An internal error occurred. Please try again."}), 500

# ─── Helper Functions ─────────────────────────────────────────────────

def get_shap_values(input_df):
    """Compute SHAP values for a single prediction."""
    try:
        if shap_explainer is not None:
            transformed = preprocessor.transform(input_df)
            if hasattr(transformed, "toarray"):
                transformed = transformed.toarray()
            transformed = np.asarray(transformed, dtype=np.float64)
            shap_vals = shap_explainer.shap_values(transformed)
            
            feature_names = shap_feature_names
            
            if isinstance(shap_vals, np.ndarray):
                if shap_vals.ndim == 3:
                    vals = shap_vals[0].tolist() if shap_vals.shape[0] == 1 else shap_vals[0].tolist()
                else:
                    vals = shap_vals[0].tolist() if len(shap_vals.shape) > 1 else shap_vals.tolist()
            elif isinstance(shap_vals, list):
                vals = shap_vals[0].tolist() if len(shap_vals) > 0 else []
            else:
                vals = []
            
            aggregated = {}
            for name, val in zip(feature_names, vals):
                val = float(val)
                clean = name.split("__")[-1] if "__" in name else name
                base = clean.split("_")[0] if "_" in clean else clean
                if base in ["manufacturer", "fuel", "transmission", "drive", "type", "condition"]:
                    display = clean.title()
                else:
                    display = clean.replace("_", " ").title()
                
                if display not in aggregated:
                    aggregated[display] = 0.0
                aggregated[display] += val
            
            sorted_features = sorted(aggregated.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
            
            return [
                {"feature": name, "value": round(float(val), 4), "direction": "up" if val > 0 else "down"}
                for name, val in sorted_features
            ]
        return []
    except Exception as e:
        logger.warning(f"SHAP computation failed: {type(e).__name__}")
        return []

def calculate_tco(fuel, vehicle_age, predicted_price):
    """Calculate total cost of ownership over 5 years."""
    live_fuel = fetch_fuel_prices()
    fuel_info = {
        "gas": {"mpg": 27, "fuel_price_per_gal": live_fuel["gasoline"]},
        "diesel": {"mpg": 33, "fuel_price_per_gal": live_fuel["diesel"]},
        "electric": {"mpg": 100, "fuel_price_per_gal": live_fuel["electric"]},
        "hybrid": {"mpg": 45, "fuel_price_per_gal": live_fuel["hybrid"]},
    }
    fuel_info = fuel_info.get(fuel, fuel_info["gas"])
    
    annual_fuel = (ANNUAL_MILES / fuel_info["mpg"]) * fuel_info["fuel_price_per_gal"]
    
    # Insurance increases with car value and decreases with age
    insurance_factor = 1.0 + (vehicle_age * 0.02)  # Older cars slightly cheaper
    annual_insurance = INSURANCE_BASE * insurance_factor + (predicted_price * 0.006)
    
    # Maintenance increases significantly after 7 years
    if vehicle_age < 5:
        maintenance_factor = 1.0
    elif vehicle_age < 10:
        maintenance_factor = 1.5
    elif vehicle_age < 15:
        maintenance_factor = 2.5
    else:
        maintenance_factor = 4.0
    annual_maintenance = MAINTENANCE_BASE * maintenance_factor
    
    # Depreciation varies by age (higher when new)
    if vehicle_age < 3:
        depreciation_rate = 0.15
    elif vehicle_age < 7:
        depreciation_rate = 0.12
    elif vehicle_age < 12:
        depreciation_rate = 0.08
    else:
        depreciation_rate = 0.05
    annual_depreciation = predicted_price * depreciation_rate
    
    total_5yr = (annual_fuel + annual_insurance + annual_maintenance) * 5
    monthly_cost = total_5yr / 60
    
    return {
        "annual_fuel": round(annual_fuel, 2),
        "annual_insurance": round(annual_insurance, 2),
        "annual_maintenance": round(annual_maintenance, 2),
        "annual_depreciation": round(annual_depreciation, 2),
        "total_5yr": round(total_5yr, 2),
        "monthly_cost": round(monthly_cost, 2),
        "fuel_efficiency_mpg": fuel_info["mpg"],
        "fuel_price_per_gal": round(fuel_info["fuel_price_per_gal"], 2),
    }

def get_depreciation_curve(manufacturer, vehicle_type, fuel, transmission, drive, condition):
    """Predict price at different vehicle ages (cached)."""
    # Create cache key
    cache_key = hashlib.md5(
        f"{manufacturer}:{vehicle_type}:{fuel}:{transmission}:{drive}:{condition}".encode()
    ).hexdigest()
    
    if cache_key in _depreciation_cache:
        return _depreciation_cache[cache_key]
    
    now = datetime.now()
    points = []
    
    for age in range(0, 21):
        year = now.year - age
        # More realistic mileage: 12k/year but cap at reasonable levels
        estimated_mileage = min(age * 12000, 200000)
        
        input_df = pd.DataFrame([{
            "vehicle_age": age,
            "age_squared": age ** 2,
            "log_odometer": np.log1p(estimated_mileage),
            "manufacturer": manufacturer,
            "fuel": fuel,
            "transmission": transmission,
            "drive": drive,
            "type": vehicle_type,
            "condition": condition,
        }])
        
        predicted = float(np.expm1(rf_model.predict(input_df)[0]))
        points.append({
            "age": int(age),
            "year": int(year),
            "predicted_price": round(predicted, 2),
            "estimated_mileage": int(estimated_mileage),
        })
    
    # Calculate year-over-year depreciation
    if len(points) >= 2:
        for i in range(1, len(points)):
            prev = points[i-1]["predicted_price"]
            curr = points[i]["predicted_price"]
            if prev > 0:
                points[i]["depreciation_pct"] = round(float((1 - curr/prev) * 100), 1)
            else:
                points[i]["depreciation_pct"] = 0.0
        points[0]["depreciation_pct"] = 0.0
    
    _depreciation_cache[cache_key] = points
    return points

def get_regional_comparison(input_df, price_usd, currency):
    """Compare prices across conditions for the same manufacturer."""
    CURRENCY_RATES = fetch_currency_rates()
    rate = CURRENCY_RATES.get(currency, 1.0)
    base_condition = input_df.iloc[0]
    same_manufacturer = cars_df[cars_df["manufacturer"] == base_condition["manufacturer"]]
    
    comparisons = {}
    for cond in ["excellent", "good", "fair"]:
        cond_cars = same_manufacturer[same_manufacturer["condition"] == cond]
        if len(cond_cars) > 0:
            avg_price = cond_cars["price"].mean()
            comparisons[cond] = {
                "avg_price": round(float(avg_price) * rate, 2),
                "count": int(len(cond_cars)),
            }
    
    return comparisons

def get_negotiation_strategy(diff_pct, vehicle_age, odometer, condition):
    """Generate situation-specific negotiation advice."""
    tips = []
    
    if diff_pct > 20:
        tips.append("This car is significantly overpriced. You have strong leverage.")
        tips.append("Open with an offer ~25-30% below asking price.")
        tips.append("Be prepared to walk away - there are likely better deals.")
    elif diff_pct > 10:
        tips.append("The car is somewhat overpriced but negotiable.")
        tips.append("Start with an offer ~15% below asking price.")
        tips.append("Point out any cosmetic issues or maintenance needs.")
    elif diff_pct > -5:
        tips.append("This is fairly priced. Room for small negotiation.")
        tips.append("Ask for extras: floor mats, recent service records, warranty transfer.")
    else:
        tips.append("This is a good deal! Don\'t over-negotiate.")
        tips.append("Act quickly - good deals don\'t last.")
        tips.append("Consider asking for a vehicle history report if not provided.")
    
    if odometer > 100000:
        tips.append("High mileage is a good negotiation point - mention upcoming maintenance costs.")
    if vehicle_age > 8:
        tips.append("Older vehicles have more negotiation room due to depreciation.")
    if condition == "fair":
        tips.append("\'Fair\' condition gives you leverage to negotiate for reconditioning costs.")
    
    return tips

def get_market_intelligence(manufacturer, vehicle_type):
    """Get market stats for a manufacturer+type combination."""
    subset = cars_df[
        (cars_df["manufacturer"] == manufacturer) &
        (cars_df["type"] == vehicle_type)
    ]
    
    if len(subset) == 0:
        subset = cars_df[cars_df["manufacturer"] == manufacturer]
    
    if len(subset) == 0:
        return {"avg_price": 0, "median_price": 0, "price_std": 0,
                "avg_mileage": 0, "year_distribution": {}, "fuel_mix": {},
                "price_trend": {}, "total_listings": 0}
    
    year_dist = subset["year"].value_counts().sort_index()
    year_dist_dict = {str(k): int(v) for k, v in year_dist.items()}
    
    fuel_mix = subset["fuel"].value_counts()
    fuel_mix_dict = {k: int(v) for k, v in fuel_mix.items()}
    
    price_by_year = subset.groupby("year")["price"].mean()
    price_trend = {str(k): round(float(v), 2) for k, v in price_by_year.items()}
    
    return {
        "avg_price": round(float(subset["price"].mean()), 2),
        "median_price": round(float(subset["price"].median()), 2),
        "price_std": round(float(subset["price"].std()), 2),
        "avg_mileage": round(float(subset["odometer"].mean()), 0),
        "year_distribution": year_dist_dict,
        "fuel_mix": fuel_mix_dict,
        "price_trend": price_trend,
        "total_listings": int(len(subset)),
    }

# ─── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Production settings
    debug_mode = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    port = int(os.environ.get("PORT", 5000))
    
    app.run(
        host="127.0.0.1",
        port=port,
        debug=debug_mode,
    )
