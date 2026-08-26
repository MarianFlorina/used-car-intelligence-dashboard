"""
Live Data Module
Fetches real-time currency exchange rates and fuel prices from free APIs.
Caches results to minimize API calls.
"""
import time
import requests

# Cache settings
_cache = {}
CACHE_TTL = 3600  # 1 hour


def _is_cache_valid(key):
    if key in _cache:
        ts, _ = _cache[key]
        if time.time() - ts < CACHE_TTL:
            return True
    return False


def _set_cache(key, value):
    _cache[key] = (time.time(), value)


def _get_cache(key):
    if key in _cache:
        _, val = _cache[key]
        return val
    return None


# ─── Currency Exchange Rates ──────────────────────────────────────────

def fetch_currency_rates(base="USD"):
    """
    Fetch live exchange rates from Frankfurter API (free, no key required).
    Sources rates from 84 central banks.
    """
    cache_key = f"currency_rates_{base}"
    if _is_cache_valid(cache_key):
        return _get_cache(cache_key)

    try:
        url = f"https://api.frankfurter.dev/v2/rates?base={base}&quotes=USD,EUR,GBP,INR"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        # Frankfurter v2 returns a list: [{"date": "...", "base": "USD", "quote": "EUR", "rate": 0.85}, ...]
        result = {"USD": 1.0}

        if isinstance(data, list):
            for item in data:
                quote = item.get("quote", "")
                rate = item.get("rate", 0)
                if quote and rate:
                    result[quote] = rate
        elif isinstance(data, dict):
            rates = data.get("rates", {})
            if isinstance(rates, dict):
                for currency, rate in rates.items():
                    result[currency] = rate

        _set_cache(cache_key, result)
        return result

    except Exception as e:
        print(f"Currency API error: {e}")
        return {
            "USD": 1.0,
            "INR": 83.0,
            "EUR": 0.92,
            "GBP": 0.78,
        }


def get_last_updated(base="USD"):
    """Get the last update timestamp from the Frankfurter API."""
    try:
        url = f"https://api.frankfurter.dev/v2/rates?base={base}&quotes=USD"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and data:
            return data[0].get("date", "unknown")
        return data.get("date", "unknown")
    except Exception:
        return "unknown"


# ─── Fuel Prices ──────────────────────────────────────────────────────

def fetch_fuel_prices():
    """
    Fetch live US gasoline and diesel prices from EIA API (free, requires key).
    Falls back to national averages if API fails.

    To get a free EIA API key:
    1. Go to https://www.eia.gov/opendata/
    2. Click "Register"
    3. Enter your email
    4. You'll receive the key immediately

    Set EIA_API_KEY environment variable or edit the key below.
    """
    cache_key = "fuel_prices"
    if _is_cache_valid(cache_key):
        return _get_cache(cache_key)

    import os
    eia_key = os.environ.get("EIA_API_KEY", "")

    if eia_key:
        try:
            # Regular gasoline price (national average, $/gallon)
            gas_url = (
                f"https://api.eia.gov/v2/petroleum/pri/gnd/data/"
                f"?api_key={eia_key}"
                f"&data[]=value"
                f"&facets[product][]=EPMR"
                f"&sort[0][column]=period"
                f"&sort[0][direction]=desc"
                f"&length=1"
            )
            gas_resp = requests.get(gas_url, timeout=10)
            gas_data = gas_resp.json()

            gas_price = None
            diesel_price = None

            if gas_data.get("response", {}).get("data"):
                gas_price = float(gas_data["response"]["data"][0]["value"])

            # Diesel price
            diesel_url = (
                f"https://api.eia.gov/v2/petroleum/pri/gnd/data/"
                f"?api_key={eia_key}"
                f"&data[]=value"
                f"&facets[product][]=EPD2D"
                f"&sort[0][column]=period"
                f"&sort[0][direction]=desc"
                f"&length=1"
            )
            diesel_resp = requests.get(diesel_url, timeout=10)
            diesel_data = diesel_resp.json()

            if diesel_data.get("response", {}).get("data"):
                diesel_price = float(diesel_data["response"]["data"][0]["value"])

            if gas_price or diesel_price:
                result = {
                    "gasoline": gas_price or 3.50,
                    "diesel": diesel_price or 3.80,
                    "electric": 0.12,  # $/kWh national average
                    "hybrid": gas_price or 3.50,
                    "source": "EIA (live)",
                    "last_updated": get_last_updated(),
                }
                _set_cache(cache_key, result)
                return result

        except Exception as e:
            print(f"EIA API error: {e}")

    # Fallback: return national average estimates
    result = {
        "gasoline": 3.50,
        "diesel": 3.80,
        "electric": 0.12,
        "hybrid": 3.50,
        "source": "National average (fallback)",
        "last_updated": "estimated",
    }
    _set_cache(cache_key, result)
    return result
