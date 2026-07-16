"""
app.py
------
Main Streamlit application for the Weather Prediction Web Application.

Pages (via sidebar navigation):
    - Home / Current Weather
    - Historical Weather
    - Forecast
    - AI Prediction (LSTM)
    - About

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, datetime, timedelta

from utils import (
    reverse_geocode, search_city, get_current_and_forecast,
    get_air_quality, get_historical_weather, wmo_to_text, safe_round,
)
from preprocess import (
    raw_to_dataframe, handle_missing_values, remove_outliers,
    engineer_features, FEATURE_COLUMNS,
)
from model import load_lstm_model, load_scaler, predict_next_temperature, model_files_exist

# ---------------------------------------------------------------------------
# Page configuration + theme
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Skyline — Weather Prediction",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Design tokens: a "storm-to-clear-sky" palette rather than default Streamlit
# blue/red. Deep slate-indigo for night/storm states, a warm amber accent for
# the sun, and a clean paper-white surface for readability.
PALETTE = {
    "bg": "#0B1220",
    "surface": "#121B2E",
    "surface_alt": "#1A2740",
    "text": "#EAF0FB",
    "text_dim": "#93A3C4",
    "accent": "#F2A65A",   # sunrise amber — the signature accent
    "accent2": "#5AC8F2",  # sky blue
    "good": "#5FD3A0",
    "warn": "#F2A65A",
    "bad": "#F2685A",
}

CUSTOM_CSS = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"]  {{
        font-family: 'Inter', sans-serif;
    }}
    h1, h2, h3, .signature-font {{
        font-family: 'Space Grotesk', sans-serif !important;
    }}

    .stApp {{
        background: linear-gradient(180deg, {PALETTE['bg']} 0%, #0E1830 100%);
        color: {PALETTE['text']};
    }}

    section[data-testid="stSidebar"] {{
        background-color: {PALETTE['surface']};
        border-right: 1px solid rgba(255,255,255,0.06);
    }}

    .weather-card {{
        background: {PALETTE['surface']};
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.8rem;
    }}
    .weather-card .label {{
        color: {PALETTE['text_dim']};
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.2rem;
    }}
    .weather-card .value {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.6rem;
        font-weight: 700;
        color: {PALETTE['text']};
    }}
    .hero-temp {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 4.2rem;
        font-weight: 700;
        line-height: 1;
        color: {PALETTE['text']};
    }}
    .hero-condition {{
        color: {PALETTE['accent']};
        font-size: 1.1rem;
        font-weight: 600;
    }}
    .divider-line {{
        border: none;
        border-top: 1px solid rgba(255,255,255,0.08);
        margin: 1.2rem 0;
    }}
    .eyebrow {{
        color: {PALETTE['accent2']};
        font-size: 0.8rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        font-weight: 600;
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PLOTLY_TEMPLATE = dict(
    layout=go.Layout(
        paper_bgcolor=PALETTE["surface"],
        plot_bgcolor=PALETTE["surface"],
        font=dict(color=PALETTE["text"], family="Inter, sans-serif"),
        colorway=[PALETTE["accent"], PALETTE["accent2"], PALETTE["good"], PALETTE["bad"]],
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)"),
        margin=dict(t=40, l=10, r=10, b=10),
    )
)


def apply_theme(fig: go.Figure, **overrides) -> go.Figure:
    """Apply the shared dark theme to a Plotly figure, then layer on any
    chart-specific overrides (e.g. a secondary y-axis, custom height, or a
    legend position).

    Overrides take precedence over the base theme for any overlapping key
    (like 'yaxis'), since dict.update() replaces rather than duplicates.
    Calling fig.update_layout(**base_dict, yaxis=...) directly, instead of
    through this helper, raises "got multiple values for keyword argument"
    whenever the override also sets a key already present in the base theme
    -- this helper merges the two dicts first so that never happens.
    """
    layout_kwargs = PLOTLY_TEMPLATE["layout"].to_plotly_json()
    layout_kwargs.update(overrides)
    fig.update_layout(**layout_kwargs)
    return fig

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

if "lat" not in st.session_state:
    st.session_state.lat = 28.6139  # default: New Delhi
if "lon" not in st.session_state:
    st.session_state.lon = 77.2090
if "place_name" not in st.session_state:
    st.session_state.place_name = "New Delhi, India"


def set_location(lat: float, lon: float, name: str = None):
    st.session_state.lat = lat
    st.session_state.lon = lon
    st.session_state.place_name = name or reverse_geocode(lat, lon)


# ---------------------------------------------------------------------------
# Sidebar — navigation + location controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        f"<div class='eyebrow'>Skyline</div>"
        f"<h2 style='margin-top:0.2rem;'>Weather Prediction</h2>",
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigate",
        ["Current Weather", "Historical Weather", "Forecast", "AI Prediction", "About"],
        label_visibility="collapsed",
    )

    st.markdown("<hr class='divider-line'>", unsafe_allow_html=True)
    st.markdown("<div class='eyebrow'>Location</div>", unsafe_allow_html=True)

    # --- GPS detection ---
    try:
        from streamlit_geolocation import streamlit_geolocation
        st.caption("Tap to detect your current location:")
        geo = streamlit_geolocation()
        if geo and geo.get("latitude") and geo.get("longitude"):
            if (abs(geo["latitude"] - st.session_state.lat) > 0.001
                    or abs(geo["longitude"] - st.session_state.lon) > 0.001):
                set_location(geo["latitude"], geo["longitude"])
                st.rerun()
    except ImportError:
        st.caption("Install `streamlit-geolocation` to enable GPS detection.")

    # --- City search ---
    search_query = st.text_input("Search for a city", placeholder="e.g. Mumbai, Tokyo, Paris")
    if search_query:
        results = search_city(search_query)
        if results:
            options = {
                f"{r['name']}, {r.get('admin1', '')}, {r.get('country', '')}".replace(", ,", ",").strip(", "): r
                for r in results
            }
            choice = st.selectbox("Matching locations", list(options.keys()))
            if st.button("Use this location", use_container_width=True):
                r = options[choice]
                set_location(r["latitude"], r["longitude"], f"{r['name']}, {r.get('country', '')}")
                st.rerun()
        else:
            st.caption("No matches found.")

    st.markdown("<hr class='divider-line'>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='label' style='color:{PALETTE['text_dim']};font-size:0.8rem;'>Current location</div>"
        f"<div style='font-weight:600;'>{st.session_state.place_name}</div>"
        f"<div style='color:{PALETTE['text_dim']};font-size:0.8rem;'>"
        f"{st.session_state.lat:.4f}, {st.session_state.lon:.4f}</div>",
        unsafe_allow_html=True,
    )

lat, lon = st.session_state.lat, st.session_state.lon
place_name = st.session_state.place_name

# ---------------------------------------------------------------------------
# PAGE: Current Weather
# ---------------------------------------------------------------------------

if page == "Current Weather":
    st.markdown(f"<div class='eyebrow'>Now — {place_name}</div>", unsafe_allow_html=True)

    with st.spinner("Fetching current conditions..."):
        data = get_current_and_forecast(lat, lon, forecast_days=1)
        aqi_data = get_air_quality(lat, lon)

    if data and "current" in data:
        cur = data["current"]
        units = data.get("current_units", {})
        desc, icon = wmo_to_text(cur.get("weather_code"))

        col1, col2 = st.columns([1.4, 2])
        with col1:
            st.markdown(
                f"<div class='hero-temp'>{icon} {safe_round(cur.get('temperature_2m'))}°"
                f"{units.get('temperature_2m', 'C')}</div>"
                f"<div class='hero-condition'>{desc}</div>"
                f"<div style='color:{PALETTE['text_dim']};margin-top:0.3rem;'>"
                f"Feels like {safe_round(cur.get('apparent_temperature'))}°</div>",
                unsafe_allow_html=True,
            )
        with col2:
            c1, c2, c3 = st.columns(3)
            metrics = [
                ("Humidity", f"{safe_round(cur.get('relative_humidity_2m'))}%"),
                ("Wind Speed", f"{safe_round(cur.get('wind_speed_10m'))} km/h"),
                ("Pressure", f"{safe_round(cur.get('surface_pressure'))} hPa"),
                ("Visibility", f"{safe_round((cur.get('visibility') or 0)/1000, 1)} km"),
                ("UV Index", f"{safe_round(cur.get('uv_index'))}"),
                ("Precipitation", f"{safe_round(cur.get('precipitation'))} mm"),
            ]
            for i, (label, value) in enumerate(metrics):
                target_col = [c1, c2, c3][i % 3]
                with target_col:
                    st.markdown(
                        f"<div class='weather-card'><div class='label'>{label}</div>"
                        f"<div class='value'>{value}</div></div>",
                        unsafe_allow_html=True,
                    )

        if aqi_data and "current" in aqi_data:
            aqi = aqi_data["current"]
            st.markdown("<hr class='divider-line'>", unsafe_allow_html=True)
            st.markdown("<div class='eyebrow'>Air Quality</div>", unsafe_allow_html=True)
            a1, a2, a3 = st.columns(3)
            aqi_metrics = [
                ("US AQI", safe_round(aqi.get("us_aqi"))),
                ("PM2.5", f"{safe_round(aqi.get('pm2_5'))} µg/m³"),
                ("PM10", f"{safe_round(aqi.get('pm10'))} µg/m³"),
            ]
            for col, (label, value) in zip([a1, a2, a3], aqi_metrics):
                with col:
                    st.markdown(
                        f"<div class='weather-card'><div class='label'>{label}</div>"
                        f"<div class='value'>{value}</div></div>",
                        unsafe_allow_html=True,
                    )

        # Today's hourly trend
        if "hourly" in data:
            st.markdown("<hr class='divider-line'>", unsafe_allow_html=True)
            st.markdown("<div class='eyebrow'>Today's Hourly Trend</div>", unsafe_allow_html=True)
            hourly = data["hourly"]
            hdf = pd.DataFrame({
                "time": pd.to_datetime(hourly["time"][:24]),
                "temperature": hourly["temperature_2m"][:24],
                "rain_probability": hourly["precipitation_probability"][:24],
            })
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hdf["time"], y=hdf["temperature"], name="Temperature (°C)",
                                      mode="lines+markers", line=dict(color=PALETTE["accent"], width=3)))
            fig.add_trace(go.Bar(x=hdf["time"], y=hdf["rain_probability"], name="Rain probability (%)",
                                  yaxis="y2", opacity=0.35, marker_color=PALETTE["accent2"]))
            apply_theme(
                fig,
                yaxis=dict(title="°C", gridcolor="rgba(255,255,255,0.06)"),
                yaxis2=dict(title="%", overlaying="y", side="right", showgrid=False, range=[0, 100]),
                height=380,
                legend=dict(orientation="h", y=1.12),
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Unable to load current weather data. Please check your connection and try again.")

# ---------------------------------------------------------------------------
# PAGE: Historical Weather
# ---------------------------------------------------------------------------

elif page == "Historical Weather":
    st.markdown(f"<div class='eyebrow'>Historical Weather — {place_name}</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input("Start date", value=date.today() - timedelta(days=30),
                               max_value=date.today() - timedelta(days=5))
    with col2:
        end = st.date_input("End date", value=date.today() - timedelta(days=5),
                             max_value=date.today() - timedelta(days=5))

    if start > end:
        st.warning("Start date must be before end date.")
    else:
        with st.spinner("Fetching historical records..."):
            hist_json = get_historical_weather(lat, lon, start.isoformat(), end.isoformat())

        if hist_json and "daily" in hist_json:
            df = raw_to_dataframe(hist_json, lat, lon)
            df = handle_missing_values(df)

            if df.empty:
                st.info("No historical data available for this range.")
            else:
                latest = df.iloc[-1]
                st.markdown("<div class='eyebrow'>Summary (most recent day in range)</div>", unsafe_allow_html=True)
                cols = st.columns(4)
                summary = [
                    ("Min Temp", f"{safe_round(latest['temp_min'])}°C"),
                    ("Max Temp", f"{safe_round(latest['temp_max'])}°C"),
                    ("Avg Temp", f"{safe_round(latest['temp_mean'])}°C"),
                    ("Humidity", f"{safe_round(latest['humidity'])}%"),
                    ("Wind Speed", f"{safe_round(latest['wind_speed'])} km/h"),
                    ("Rainfall", f"{safe_round(latest['precipitation'])} mm"),
                    ("Pressure", f"{safe_round(latest['pressure'])} hPa"),
                ]
                for i, (label, value) in enumerate(summary):
                    with cols[i % 4]:
                        st.markdown(
                            f"<div class='weather-card'><div class='label'>{label}</div>"
                            f"<div class='value'>{value}</div></div>",
                            unsafe_allow_html=True,
                        )

                st.markdown("<hr class='divider-line'>", unsafe_allow_html=True)
                st.markdown("<div class='eyebrow'>Temperature Trend</div>", unsafe_allow_html=True)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df["date"], y=df["temp_max"], name="Max", line=dict(color=PALETTE["bad"])))
                fig.add_trace(go.Scatter(x=df["date"], y=df["temp_min"], name="Min", line=dict(color=PALETTE["accent2"]),
                                          fill="tonexty", fillcolor="rgba(90,200,242,0.08)"))
                fig.add_trace(go.Scatter(x=df["date"], y=df["temp_mean"], name="Mean", line=dict(color=PALETTE["accent"], dash="dot")))
                apply_theme(fig, height=380)
                st.plotly_chart(fig, use_container_width=True)

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("<div class='eyebrow'>Rainfall</div>", unsafe_allow_html=True)
                    fig2 = px.bar(df, x="date", y="precipitation")
                    fig2.update_traces(marker_color=PALETTE["accent2"])
                    apply_theme(fig2, height=300)
                    st.plotly_chart(fig2, use_container_width=True)
                with c2:
                    st.markdown("<div class='eyebrow'>Humidity</div>", unsafe_allow_html=True)
                    fig3 = px.line(df, x="date", y="humidity")
                    fig3.update_traces(line_color=PALETTE["good"])
                    apply_theme(fig3, height=300)
                    st.plotly_chart(fig3, use_container_width=True)

                # Monthly averages + correlation heatmap (only meaningful for longer ranges)
                if len(df) >= 14:
                    st.markdown("<hr class='divider-line'>", unsafe_allow_html=True)
                    c3, c4 = st.columns(2)
                    with c3:
                        st.markdown("<div class='eyebrow'>Monthly Averages</div>", unsafe_allow_html=True)
                        monthly = df.copy()
                        monthly["month"] = monthly["date"].dt.to_period("M").astype(str)
                        monthly_avg = monthly.groupby("month")[["temp_mean", "humidity", "precipitation"]].mean().reset_index()
                        fig4 = px.bar(monthly_avg, x="month", y="temp_mean")
                        fig4.update_traces(marker_color=PALETTE["accent"])
                        apply_theme(fig4, height=300)
                        st.plotly_chart(fig4, use_container_width=True)
                    with c4:
                        st.markdown("<div class='eyebrow'>Correlation Heatmap</div>", unsafe_allow_html=True)
                        corr_cols = ["temp_mean", "humidity", "pressure", "wind_speed", "precipitation"]
                        corr = df[corr_cols].corr()
                        fig5 = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
                        apply_theme(fig5, height=300)
                        st.plotly_chart(fig5, use_container_width=True)

                with st.expander("View raw data table"):
                    st.dataframe(df.drop(columns=["latitude", "longitude"]), use_container_width=True)
        else:
            st.error("Unable to load historical data for this location and date range.")

# ---------------------------------------------------------------------------
# PAGE: Forecast
# ---------------------------------------------------------------------------

elif page == "Forecast":
    st.markdown(f"<div class='eyebrow'>Forecast — {place_name}</div>", unsafe_allow_html=True)
    forecast_days = st.slider("Forecast horizon (days)", min_value=1, max_value=16, value=7)

    with st.spinner("Fetching forecast..."):
        data = get_current_and_forecast(lat, lon, forecast_days=forecast_days)

    if data and "daily" in data:
        daily = data["daily"]
        fdf = pd.DataFrame({
            "date": pd.to_datetime(daily["time"]),
            "temp_max": daily["temperature_2m_max"],
            "temp_min": daily["temperature_2m_min"],
            "rain_prob": daily["precipitation_probability_max"],
            "precipitation": daily["precipitation_sum"],
            "wind_speed": daily["wind_speed_10m_max"],
            "uv_index": daily["uv_index_max"],
            "weather_code": daily["weather_code"],
            "sunrise": daily["sunrise"],
            "sunset": daily["sunset"],
        })

        st.markdown("<div class='eyebrow'>Daily Forecast</div>", unsafe_allow_html=True)
        cols = st.columns(min(len(fdf), 7))
        for i, row in fdf.iterrows():
            if i >= len(cols):
                break
            desc, icon = wmo_to_text(row["weather_code"])
            with cols[i]:
                st.markdown(
                    f"<div class='weather-card' style='text-align:center;'>"
                    f"<div class='label'>{row['date'].strftime('%a %d %b')}</div>"
                    f"<div style='font-size:2rem;'>{icon}</div>"
                    f"<div class='value' style='font-size:1.1rem;'>{safe_round(row['temp_max'])}° / {safe_round(row['temp_min'])}°</div>"
                    f"<div style='color:{PALETTE['accent2']};font-size:0.85rem;'>💧 {safe_round(row['rain_prob'])}%</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("<hr class='divider-line'>", unsafe_allow_html=True)
        st.markdown("<div class='eyebrow'>Forecast Trend</div>", unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fdf["date"], y=fdf["temp_max"], name="Max Temp", line=dict(color=PALETTE["bad"])))
        fig.add_trace(go.Scatter(x=fdf["date"], y=fdf["temp_min"], name="Min Temp", line=dict(color=PALETTE["accent2"])))
        fig.add_trace(go.Bar(x=fdf["date"], y=fdf["rain_prob"], name="Rain probability (%)", yaxis="y2", opacity=0.3, marker_color=PALETTE["good"]))
        apply_theme(
            fig,
            yaxis=dict(title="°C"),
            yaxis2=dict(title="%", overlaying="y", side="right", showgrid=False, range=[0, 100]),
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Hourly forecast for the first selected day
        if "hourly" in data:
            st.markdown("<hr class='divider-line'>", unsafe_allow_html=True)
            st.markdown("<div class='eyebrow'>Hourly Detail (next 48h)</div>", unsafe_allow_html=True)
            hourly = data["hourly"]
            hdf = pd.DataFrame({
                "time": pd.to_datetime(hourly["time"][:48]),
                "temperature": hourly["temperature_2m"][:48],
                "wind_speed": hourly["wind_speed_10m"][:48],
            })
            fig2 = px.line(hdf, x="time", y=["temperature", "wind_speed"])
            apply_theme(fig2, height=320)
            st.plotly_chart(fig2, use_container_width=True)

        with st.expander("View raw forecast table"):
            st.dataframe(fdf, use_container_width=True)
    else:
        st.error("Unable to load forecast data.")

# ---------------------------------------------------------------------------
# PAGE: AI Prediction (LSTM)
# ---------------------------------------------------------------------------

elif page == "AI Prediction":
    st.markdown(f"<div class='eyebrow'>AI Weather Prediction — {place_name}</div>", unsafe_allow_html=True)
    st.caption(
        "This uses a trained LSTM neural network (see `train_model.py`) to predict "
        "tomorrow's mean temperature from the last 14 days of weather patterns."
    )

    if not model_files_exist():
        st.warning(
            "No trained model found yet. Train one first by running, in your terminal:\n\n"
            "`python train_model.py --lat {:.4f} --lon {:.4f} --years 5`\n\n"
            "This fetches historical data, trains the LSTM, and saves it to the `models/` folder."
            .format(lat, lon)
        )
    else:
        with st.spinner("Loading model and recent data..."):
            model = load_lstm_model()
            scaler = load_scaler()

            lookback = 14
            end = date.today() - timedelta(days=5)
            start = end - timedelta(days=lookback + 5)  # small buffer for interpolation
            hist_json = get_historical_weather(lat, lon, start.isoformat(), end.isoformat())

        if model is None or scaler is None:
            st.error("Model or scaler failed to load. Try retraining with train_model.py.")
        elif hist_json is None:
            st.error("Could not fetch the recent historical data needed for prediction.")
        else:
            df = raw_to_dataframe(hist_json, lat, lon)
            df = handle_missing_values(df)
            df = engineer_features(df)

            if len(df) < lookback:
                st.warning(f"Not enough recent data ({len(df)} days) for a {lookback}-day lookback window.")
            else:
                window = df.tail(lookback)
                try:
                    predicted_temp = predict_next_temperature(model, scaler, window, FEATURE_COLUMNS)

                    is_fallback = not 'keras' in str(type(model))
                    if is_fallback:
                        st.info("ℹ️ Running in **Fallback Mode** using Scikit-Learn (Random Forest) because TensorFlow is not installed in this environment.")

                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.markdown(
                            f"<div class='weather-card' style='text-align:center;'>"
                            f"<div class='label'>Predicted temp — tomorrow</div>"
                            f"<div class='hero-temp' style='font-size:3rem;'>{predicted_temp:.1f}°C</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    with col2:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=window["date"], y=window["temp_mean"], name="Last 14 days (actual)",
                                                  line=dict(color=PALETTE["accent2"], width=3)))
                        next_day = window["date"].max() + timedelta(days=1)
                        fig.add_trace(go.Scatter(x=[window["date"].max(), next_day],
                                                  y=[window["temp_mean"].iloc[-1], predicted_temp],
                                                  name="AI prediction", line=dict(color=PALETTE["accent"], dash="dash", width=3),
                                                  mode="lines+markers"))
                        apply_theme(fig, height=320)
                        st.plotly_chart(fig, use_container_width=True)

                    st.info(
                        "Note: this prediction is generated by a lightweight LSTM trained on a single "
                        "location's recent history. Accuracy improves with more years of training data "
                        "and is intended as a demonstration, not a substitute for official forecasts."
                    )
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

    with st.expander("How the AI model works"):
        st.markdown(
            """
The model is a stacked **LSTM (Long Short-Term Memory)** neural network trained in `train_model.py`:

1. **Data**: 5 years of daily historical weather for a chosen location, pulled from the Open-Meteo Archive API.
2. **Features**: latitude, longitude, mean temperature, humidity, pressure, wind speed, precipitation, month, season, and cyclical day-of-year encodings (sin/cos).
3. **Architecture**: `LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(16, relu) → Dense(1)`.
4. **Training**: 80/20 chronological train/test split, Adam optimizer, early stopping (patience=10), and checkpointing to keep the best weights.
5. **Evaluation**: RMSE, MAE, and R² are reported after training.
            """
        )

# ---------------------------------------------------------------------------
# PAGE: About
# ---------------------------------------------------------------------------

elif page == "About":
    st.markdown("<div class='eyebrow'>About</div>", unsafe_allow_html=True)
    st.markdown(
        """
## Skyline — Weather Prediction

A full-stack weather dashboard built with **Streamlit**, **Plotly**, and a custom
**LSTM neural network** for AI-based temperature prediction.

**Data sources**
- Current, hourly, and daily forecast: [Open-Meteo Forecast API](https://open-meteo.com/)
- Historical daily weather: Open-Meteo Archive API
- Air quality: Open-Meteo Air Quality API
- Geocoding: Open-Meteo Geocoding API + Nominatim (via Geopy) for reverse geocoding

**Tech stack**: Python, Streamlit, Pandas, NumPy, Scikit-learn, TensorFlow (LSTM),
Plotly, Requests, Geopy.

See the project `README.md` for installation and deployment instructions.
        """
    )
