"""
utils.py
--------
Shared helper functions for the Weather Prediction Web Application.

This module centralizes all external I/O so the rest of the app never talks
to third-party services directly:
    - Geocoding / reverse geocoding (via Geopy -> Nominatim)
    - Current weather (Open-Meteo)
    - Historical weather (Open-Meteo Archive API)
    - Forecast weather, hourly + daily (Open-Meteo)
    - Air quality (Open-Meteo Air Quality API)

Open-Meteo is used because it is completely free and requires no API key.
OpenWeatherMap support is included as an optional secondary source if the
user supplies an API key via Streamlit secrets (OWM_API_KEY).
"""

import requests
import streamlit as st
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from datetime import datetime, date
from typing import Optional, Tuple, Dict, Any, List

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GEOCODE_USER_AGENT = "weather_prediction_app"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
OPEN_METEO_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"

# WMO weather interpretation codes -> (description, emoji icon)
WMO_CODES: Dict[int, Tuple[str, str]] = {
    0: ("Clear sky", "☀️"),
    1: ("Mainly clear", "🌤️"),
    2: ("Partly cloudy", "⛅"),
    3: ("Overcast", "☁️"),
    45: ("Fog", "🌫️"),
    48: ("Depositing rime fog", "🌫️"),
    51: ("Light drizzle", "🌦️"),
    53: ("Moderate drizzle", "🌦️"),
    55: ("Dense drizzle", "🌧️"),
    56: ("Light freezing drizzle", "🌧️"),
    57: ("Dense freezing drizzle", "🌧️"),
    61: ("Slight rain", "🌦️"),
    63: ("Moderate rain", "🌧️"),
    65: ("Heavy rain", "🌧️"),
    66: ("Light freezing rain", "🌧️"),
    67: ("Heavy freezing rain", "🌧️"),
    71: ("Slight snow fall", "🌨️"),
    73: ("Moderate snow fall", "🌨️"),
    75: ("Heavy snow fall", "❄️"),
    77: ("Snow grains", "❄️"),
    80: ("Slight rain showers", "🌦️"),
    81: ("Moderate rain showers", "🌧️"),
    82: ("Violent rain showers", "⛈️"),
    85: ("Slight snow showers", "🌨️"),
    86: ("Heavy snow showers", "❄️"),
    95: ("Thunderstorm", "⛈️"),
    96: ("Thunderstorm with slight hail", "⛈️"),
    99: ("Thunderstorm with heavy hail", "⛈️"),
}


def wmo_to_text(code: Optional[int]) -> Tuple[str, str]:
    """Translate a WMO weather code into (description, icon). Falls back
    gracefully if the code is missing or unknown."""
    if code is None:
        return ("Unknown", "❓")
    return WMO_CODES.get(int(code), ("Unknown", "❓"))


# ---------------------------------------------------------------------------
# Geocoding
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _get_geolocator() -> Nominatim:
    """Create a single cached Nominatim geolocator instance for the app."""
    return Nominatim(user_agent=GEOCODE_USER_AGENT, timeout=10)


@st.cache_data(show_spinner=False, ttl=3600)
def reverse_geocode(lat: float, lon: float) -> str:
    """Convert latitude/longitude into a human-readable place name.

    Returns a fallback 'lat, lon' string if the lookup fails, so the UI
    never breaks even when the geocoding service is unreachable.
    """
    try:
        geolocator = _get_geolocator()
        location = geolocator.reverse((lat, lon), language="en", exactly_one=True)
        if location and location.raw.get("address"):
            addr = location.raw["address"]
            city = (
                addr.get("city")
                or addr.get("town")
                or addr.get("village")
                or addr.get("county")
                or addr.get("state")
                or "Unknown"
            )
            country = addr.get("country", "")
            return f"{city}, {country}".strip(", ")
        return f"{lat:.3f}, {lon:.3f}"
    except (GeocoderTimedOut, GeocoderServiceError, Exception):
        return f"{lat:.3f}, {lon:.3f}"


@st.cache_data(show_spinner=False, ttl=3600)
def search_city(query: str, count: int = 5) -> List[Dict[str, Any]]:
    """Search for a city by name using the Open-Meteo geocoding API.

    Returns a list of dicts with keys: name, country, admin1, latitude,
    longitude, population. Returns an empty list on failure.
    """
    if not query or len(query.strip()) < 2:
        return []
    try:
        resp = requests.get(
            OPEN_METEO_GEOCODE_URL,
            params={"name": query.strip(), "count": count, "language": "en", "format": "json"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("results", []) or []
    except requests.RequestException:
        return []


# ---------------------------------------------------------------------------
# Current + Forecast weather (Open-Meteo)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=600)
def get_current_and_forecast(lat: float, lon: float, forecast_days: int = 7) -> Optional[Dict[str, Any]]:
    """Fetch current conditions plus hourly/daily forecast from Open-Meteo.

    A single API call is used for efficiency: Open-Meteo lets us request
    'current', 'hourly', and 'daily' blocks together.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": [
            "temperature_2m", "relative_humidity_2m", "apparent_temperature",
            "precipitation", "weather_code", "surface_pressure",
            "wind_speed_10m", "visibility", "uv_index",
        ],
        "hourly": [
            "temperature_2m", "relative_humidity_2m", "precipitation_probability",
            "weather_code", "wind_speed_10m",
        ],
        "daily": [
            "temperature_2m_max", "temperature_2m_min", "precipitation_sum",
            "precipitation_probability_max", "weather_code", "wind_speed_10m_max",
            "uv_index_max", "sunrise", "sunset",
        ],
        "forecast_days": forecast_days,
        "timezone": "auto",
    }
    try:
        resp = requests.get(OPEN_METEO_FORECAST_URL, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        st.error(f"Could not fetch forecast data: {e}")
        return None


@st.cache_data(show_spinner=False, ttl=3600)
def get_air_quality(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """Fetch current air quality index (US AQI + European AQI) if available."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": ["us_aqi", "european_aqi", "pm2_5", "pm10"],
        "timezone": "auto",
    }
    try:
        resp = requests.get(OPEN_METEO_AIR_QUALITY_URL, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return None


# ---------------------------------------------------------------------------
# Historical weather (Open-Meteo Archive API)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=86400)
def get_historical_weather(
    lat: float, lon: float, start_date: str, end_date: str
) -> Optional[Dict[str, Any]]:
    """Fetch daily historical weather between start_date and end_date
    (both 'YYYY-MM-DD' strings) from the Open-Meteo Archive API.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": [
            "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
            "precipitation_sum", "wind_speed_10m_max", "relative_humidity_2m_mean",
            "surface_pressure_mean", "sunrise", "sunset",
        ],
        "timezone": "auto",
    }
    try:
        resp = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=20)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        st.error(f"Could not fetch historical data: {e}")
        return None


# ---------------------------------------------------------------------------
# Small formatting helpers
# ---------------------------------------------------------------------------

def celsius_to_fahrenheit(c: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return c * 9 / 5 + 32


def safe_round(value: Any, digits: int = 1) -> Any:
    """Round a numeric value if possible, otherwise return it unchanged."""
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return value


def get_season(month: int, hemisphere: str = "N") -> str:
    """Return the meteorological season for a given month (1-12).

    Uses the Northern Hemisphere by default; pass hemisphere='S' for the
    Southern Hemisphere (seasons offset by 6 months).
    """
    seasons_n = {
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Autumn", 10: "Autumn", 11: "Autumn",
    }
    seasons_s = {
        12: "Summer", 1: "Summer", 2: "Summer",
        3: "Autumn", 4: "Autumn", 5: "Autumn",
        6: "Winter", 7: "Winter", 8: "Winter",
        9: "Spring", 10: "Spring", 11: "Spring",
    }
    table = seasons_n if hemisphere == "N" else seasons_s
    return table.get(month, "Unknown")
