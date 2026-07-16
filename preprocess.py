"""
preprocess.py
-------------
Data cleaning, feature engineering, and scaling utilities used to turn raw
historical weather API responses into a model-ready dataset for the LSTM.

Pipeline overview:
    1. raw_to_dataframe   -> convert Open-Meteo JSON into a tidy DataFrame
    2. handle_missing_values -> interpolate / fill gaps
    3. remove_outliers    -> IQR-based clipping of extreme values
    4. engineer_features  -> add month, day-of-year, season, cyclical encodings
    5. scale_features     -> StandardScaler, fit on train split only
    6. train_test_split_series -> chronological 80/20 split (no shuffling,
       since this is a time series)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
from sklearn.preprocessing import StandardScaler

from utils import get_season


# ---------------------------------------------------------------------------
# Step 1: Raw JSON -> DataFrame
# ---------------------------------------------------------------------------

def raw_to_dataframe(historical_json: Dict[str, Any], lat: float, lon: float) -> pd.DataFrame:
    """Convert an Open-Meteo Archive API JSON response into a tidy DataFrame.

    Each row is one day, with columns for all requested daily variables plus
    the latitude/longitude of the query location (useful as model features
    when training on multiple locations).
    """
    daily = historical_json.get("daily", {})
    if not daily or "time" not in daily:
        return pd.DataFrame()

    df = pd.DataFrame({
        "date": pd.to_datetime(daily.get("time", [])),
        "temp_max": daily.get("temperature_2m_max", []),
        "temp_min": daily.get("temperature_2m_min", []),
        "temp_mean": daily.get("temperature_2m_mean", []),
        "precipitation": daily.get("precipitation_sum", []),
        "wind_speed": daily.get("wind_speed_10m_max", []),
        "humidity": daily.get("relative_humidity_2m_mean", []),
        "pressure": daily.get("surface_pressure_mean", []),
    })
    df["latitude"] = lat
    df["longitude"] = lon
    return df


# ---------------------------------------------------------------------------
# Step 2: Missing values
# ---------------------------------------------------------------------------

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill gaps in the time series.

    Numeric columns are linearly interpolated (appropriate for smoothly
    varying signals like temperature/pressure), with any remaining edge
    NaNs forward/backward filled.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit_direction="both")
    df[numeric_cols] = df[numeric_cols].ffill().bfill()
    return df


# ---------------------------------------------------------------------------
# Step 3: Outlier removal
# ---------------------------------------------------------------------------

def remove_outliers(df: pd.DataFrame, columns: List[str], iqr_multiplier: float = 3.0) -> pd.DataFrame:
    """Clip extreme outliers in the given columns using the IQR method.

    We clip rather than drop rows, since dropping would break the temporal
    continuity that the LSTM depends on.
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        df[col] = df[col].clip(lower=lower, upper=upper)
    return df


# ---------------------------------------------------------------------------
# Step 4: Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar-based and cyclical features that help the LSTM learn
    seasonal patterns:
        - month, day_of_year
        - season (categorical, encoded as an integer)
        - sin/cos encodings of day-of-year (captures cyclical seasonality
          without an artificial jump between Dec 31 and Jan 1)
    """
    df = df.copy()
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["season"] = df["month"].apply(lambda m: get_season(m))

    season_map = {"Winter": 0, "Spring": 1, "Summer": 2, "Autumn": 3}
    df["season_encoded"] = df["season"].map(season_map)

    days_in_year = 365.25
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / days_in_year)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / days_in_year)
    return df


# ---------------------------------------------------------------------------
# Step 5: Scaling
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "latitude", "longitude", "temp_mean", "humidity", "pressure",
    "wind_speed", "precipitation", "month", "season_encoded",
    "doy_sin", "doy_cos",
]
TARGET_COLUMN = "temp_mean"


def scale_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: List[str] = None
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Fit a StandardScaler on the training split ONLY, then transform both
    the train and test splits. Fitting only on train avoids data leakage
    from the future into the training process.
    """
    feature_cols = feature_cols or FEATURE_COLUMNS
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df[feature_cols])
    test_scaled = scaler.transform(test_df[feature_cols])
    return train_scaled, test_scaled, scaler


# ---------------------------------------------------------------------------
# Step 6: Chronological train/test split
# ---------------------------------------------------------------------------

def train_test_split_series(df: pd.DataFrame, test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a time-ordered DataFrame into train/test WITHOUT shuffling.

    Shuffling a time series before splitting would leak future information
    into the training set, so we always take the earliest (1 - test_ratio)
    fraction of rows for training and the most recent rows for testing.
    """
    df = df.sort_values("date").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_ratio))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def create_sequences(data: np.ndarray, target: np.ndarray, lookback: int = 14) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a 2D array of scaled features into overlapping sequences for
    the LSTM, where each input sample is `lookback` consecutive days and the
    label is the target value on the day immediately following the window.
    """
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(target[i])
    return np.array(X), np.array(y)


def full_preprocessing_pipeline(
    historical_json: Dict[str, Any], lat: float, lon: float, lookback: int = 14
) -> Dict[str, Any]:
    """Run the entire preprocessing pipeline end-to-end and return everything
    train_model.py needs: sequences, scaler, and the cleaned DataFrame.
    """
    df = raw_to_dataframe(historical_json, lat, lon)
    if df.empty:
        raise ValueError("No historical data returned for this location/date range.")

    df = handle_missing_values(df)
    df = remove_outliers(df, ["temp_max", "temp_min", "temp_mean", "wind_speed", "precipitation", "pressure"])
    df = engineer_features(df)

    train_df, test_df = train_test_split_series(df, test_ratio=0.2)
    train_scaled, test_scaled, scaler = scale_features(train_df, test_df)

    train_target = train_df[TARGET_COLUMN].values
    test_target = test_df[TARGET_COLUMN].values

    X_train, y_train = create_sequences(train_scaled, train_target, lookback)
    X_test, y_test = create_sequences(test_scaled, test_target, lookback)

    return {
        "X_train": X_train, "y_train": y_train,
        "X_test": X_test, "y_test": y_test,
        "scaler": scaler, "clean_df": df,
        "feature_columns": FEATURE_COLUMNS,
    }
