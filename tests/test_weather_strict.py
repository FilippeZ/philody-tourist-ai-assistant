"""
tests/test_weather_strict.py

Unit tests verifying strict Live OpenWeatherMap API enforcement in tools/weather.py:
1. Missing OPENWEATHER_API_KEY raises ValueError immediately.
2. Network errors / timeouts / 5xx raise HTTPException(503) with the required error message.
3. Returned data does not contain 'is_mock'.
4. WeatherResponse Pydantic model validation without 'is_mock'.
"""

import os
import unittest
from unittest.mock import patch, MagicMock
import requests
from fastapi import HTTPException

from tools.weather import get_current_weather, WeatherResponse


class TestStrictWeatherFetching(unittest.TestCase):
    """Test suite for strict OpenWeather live API enforcement."""

    def test_missing_api_key_raises_value_error(self):
        """When OPENWEATHER_API_KEY is missing/empty, ValueError is raised immediately."""
        with patch.dict(os.environ, {"OPENWEATHER_API_KEY": ""}):
            with self.assertRaises(ValueError) as ctx:
                get_current_weather(city="Athens")
            self.assertIn("OPENWEATHER_API_KEY is missing", str(ctx.exception))

    def test_network_failure_raises_http_503(self):
        """When network or HTTP request fails, HTTPException(503) is raised."""
        with patch.dict(os.environ, {"OPENWEATHER_API_KEY": "dummy_test_key"}):
            with patch("requests.get", side_effect=requests.exceptions.ConnectionError("Connection refused")):
                with self.assertRaises(HTTPException) as ctx:
                    get_current_weather(city="Athens")
                self.assertEqual(ctx.exception.status_code, 503)
                self.assertIn("Weather API unavailable", ctx.exception.detail)

    def test_http_4xx_5xx_raises_http_503(self):
        """When OpenWeather returns an error response (e.g. 401 unauthorized or 500), HTTPException(503) is raised."""
        with patch.dict(os.environ, {"OPENWEATHER_API_KEY": "dummy_test_key"}):
            mock_resp = MagicMock()
            mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("401 Client Error: Unauthorized")
            with patch("requests.get", return_value=mock_resp):
                with self.assertRaises(HTTPException) as ctx:
                    get_current_weather(city="Athens")
                self.assertEqual(ctx.exception.status_code, 503)
                self.assertIn("Weather API unavailable", ctx.exception.detail)

    def test_successful_response_excludes_is_mock_flag(self):
        """Successful live API calls return normalized data with NO 'is_mock' flag."""
        sample_api_json = {
            "name": "Athens",
            "weather": [{"main": "Clear", "description": "αίθριος ουρανός"}],
            "main": {"temp": 23.5, "feels_like": 24.0, "humidity": 45},
            "rain": {"1h": 0.0},
        }
        with patch.dict(os.environ, {"OPENWEATHER_API_KEY": "dummy_test_key"}):
            mock_resp = MagicMock()
            mock_resp.raise_for_status.return_value = None
            mock_resp.json.return_value = sample_api_json
            with patch("requests.get", return_value=mock_resp):
                data = get_current_weather(city="Athens")
                self.assertNotIn("is_mock", data)
                self.assertEqual(data["city"], "Athens")
                self.assertEqual(data["temperature_c"], 23.5)
                self.assertEqual(data["condition"], "Clear")
                self.assertFalse(data["rain_expected"])
                self.assertFalse(data["is_indoor_recommended"])

                # Validate Pydantic model
                model = WeatherResponse(**data)
                self.assertFalse(hasattr(model, "is_mock"))
                self.assertEqual(model.city, "Athens")


if __name__ == "__main__":
    unittest.main()
