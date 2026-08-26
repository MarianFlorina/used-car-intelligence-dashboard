"""
Tests for Used Car Intelligence Dashboard
Run with: python -m pytest test_app.py -v
"""
import sys
import os
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from app import (
    validate_prediction_input,
    VALID_FUELS,
    VALID_TRANSMISSIONS,
    VALID_DRIVES,
    VALID_TYPES,
    VALID_CONDITIONS,
    VALID_CURRENCIES,
)

# ─── Input Validation Tests ──────────────────────────────────────────

class TestValidation:
    """Test input validation logic."""
    
    def test_valid_input(self):
        data = {
            "year": 2018,
            "odometer": 50000,
            "manufacturer": "Toyota",
            "fuel": "gas",
            "transmission": "automatic",
            "drive": "fwd",
            "type": "sedan",
            "condition": "good",
            "currency": "USD",
        }
        assert validate_prediction_input(data) is None
    
    def test_missing_required_field(self):
        data = {"year": 2018, "odometer": 50000}
        result = validate_prediction_input(data)
        assert result is not None
        assert "Missing required field" in result
    
    def test_invalid_year(self):
        data = {
            "year": 1800,
            "odometer": 50000,
            "manufacturer": "Toyota",
            "fuel": "gas",
            "transmission": "automatic",
            "drive": "fwd",
            "type": "sedan",
            "condition": "good",
        }
        result = validate_prediction_input(data)
        assert result is not None
        assert "Year must be between" in result
    
    def test_invalid_odometer(self):
        data = {
            "year": 2018,
            "odometer": 2000000,
            "manufacturer": "Toyota",
            "fuel": "gas",
            "transmission": "automatic",
            "drive": "fwd",
            "type": "sedan",
            "condition": "good",
        }
        result = validate_prediction_input(data)
        assert result is not None
        assert "Odometer must be between" in result
    
    def test_invalid_manufacturer(self):
        data = {
            "year": 2018,
            "odometer": 50000,
            "manufacturer": "Toyota123",
            "fuel": "gas",
            "transmission": "automatic",
            "drive": "fwd",
            "type": "sedan",
            "condition": "good",
        }
        result = validate_prediction_input(data)
        assert result is not None
        assert "letters" in result
    
    def test_invalid_fuel(self):
        data = {
            "year": 2018,
            "odometer": 50000,
            "manufacturer": "Toyota",
            "fuel": "banana",
            "transmission": "automatic",
            "drive": "fwd",
            "type": "sedan",
            "condition": "good",
        }
        result = validate_prediction_input(data)
        assert result is not None
        assert "Invalid fuel type" in result
    
    def test_invalid_transmission(self):
        data = {
            "year": 2018,
            "odometer": 50000,
            "manufacturer": "Toyota",
            "fuel": "gas",
            "transmission": "cvt",
            "drive": "fwd",
            "type": "sedan",
            "condition": "good",
        }
        result = validate_prediction_input(data)
        assert result is not None
        assert "Invalid transmission" in result
    
    def test_invalid_drive(self):
        data = {
            "year": 2018,
            "odometer": 50000,
            "manufacturer": "Toyota",
            "fuel": "gas",
            "transmission": "automatic",
            "drive": "awd",
            "type": "sedan",
            "condition": "good",
        }
        result = validate_prediction_input(data)
        assert result is not None
        assert "Invalid drive type" in result
    
    def test_invalid_type(self):
        data = {
            "year": 2018,
            "odometer": 50000,
            "manufacturer": "Toyota",
            "fuel": "gas",
            "transmission": "automatic",
            "drive": "fwd",
            "type": "coupe",
            "condition": "good",
        }
        result = validate_prediction_input(data)
        assert result is not None
        assert "Invalid vehicle type" in result
    
    def test_invalid_condition(self):
        data = {
            "year": 2018,
            "odometer": 50000,
            "manufacturer": "Toyota",
            "fuel": "gas",
            "transmission": "automatic",
            "drive": "fwd",
            "type": "sedan",
            "condition": "mint",
        }
        result = validate_prediction_input(data)
        assert result is not None
        assert "Invalid condition" in result
    
    def test_invalid_currency(self):
        data = {
            "year": 2018,
            "odometer": 50000,
            "manufacturer": "Toyota",
            "fuel": "gas",
            "transmission": "automatic",
            "drive": "fwd",
            "type": "sedan",
            "condition": "good",
            "currency": "XYZ",
        }
        result = validate_prediction_input(data)
        assert result is not None
        assert "Invalid currency" in result
    
    def test_negative_listed_price(self):
        data = {
            "year": 2018,
            "odometer": 50000,
            "manufacturer": "Toyota",
            "fuel": "gas",
            "transmission": "automatic",
            "drive": "fwd",
            "type": "sedan",
            "condition": "good",
            "listed_price": -1000,
        }
        result = validate_prediction_input(data)
        assert result is not None
        assert "positive" in result
    
    def test_empty_manufacturer(self):
        data = {
            "year": 2018,
            "odometer": 50000,
            "manufacturer": "",
            "fuel": "gas",
            "transmission": "automatic",
            "drive": "fwd",
            "type": "sedan",
            "condition": "good",
        }
        result = validate_prediction_input(data)
        assert result is not None
        assert "required" in result

# ─── Constants Tests ─────────────────────────────────────────────────

class TestConstants:
    """Test that validation constants are properly defined."""
    
    def test_valid_fuels(self):
        assert "gas" in VALID_FUELS
        assert "diesel" in VALID_FUELS
        assert "electric" in VALID_FUELS
        assert "hybrid" in VALID_FUELS
    
    def test_valid_transmissions(self):
        assert "automatic" in VALID_TRANSMISSIONS
        assert "manual" in VALID_TRANSMISSIONS
    
    def test_valid_drives(self):
        assert "fwd" in VALID_DRIVES
        assert "rwd" in VALID_DRIVES
        assert "4wd" in VALID_DRIVES
    
    def test_valid_types(self):
        assert "sedan" in VALID_TYPES
        assert "suv" in VALID_TYPES
        assert "truck" in VALID_TYPES
    
    def test_valid_conditions(self):
        assert "excellent" in VALID_CONDITIONS
        assert "good" in VALID_CONDITIONS
        assert "fair" in VALID_CONDITIONS
    
    def test_valid_currencies(self):
        assert "USD" in VALID_CURRENCIES
        assert "INR" in VALID_CURRENCIES
        assert "EUR" in VALID_CURRENCIES
        assert "GBP" in VALID_CURRENCIES

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
