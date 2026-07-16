"""
model.py
--------
Defines the LSTM architecture used for temperature prediction, plus helper
functions to train, evaluate, save, load, and run inference with the model.

The model predicts the next day's mean temperature given a `lookback`-day
window of engineered weather features (see preprocess.py for the feature
list: latitude, longitude, temp_mean, humidity, pressure, wind_speed,
precipitation, month, season_encoded, doy_sin, doy_cos).
"""

import os
import numpy as np
import joblib
from typing import Tuple, Dict, Any

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# TensorFlow is imported lazily inside functions where needed so that parts
# of the app which don't need training (e.g. simple UI pages) can still run
# even in environments where TensorFlow is slow to import or unavailable.


MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_weather_model.h5")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")


def build_lstm_model(input_shape: Tuple[int, int]):
    """Construct the LSTM architecture.

    Architecture:
        LSTM(64, return_sequences=True) -> Dropout(0.2)
        LSTM(32)                        -> Dropout(0.2)
        Dense(16, relu)
        Dense(1)  # predicted temperature

    input_shape = (lookback_days, num_features)
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    return model


def train_lstm(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    epochs: int = 100, batch_size: int = 16,
) -> Tuple[Any, Dict[str, list]]:
    """Train the LSTM with early stopping and checkpointing.

    Early stopping halts training once validation loss stops improving
    (patience=10), and ModelCheckpoint keeps only the best-performing
    weights, which protects against overfitting on a limited amount of
    historical data.
    """
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

    os.makedirs(MODEL_DIR, exist_ok=True)
    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))

    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True, verbose=0)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, checkpoint],
        verbose=1,
    )
    return model, history.history


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """Compute RMSE, MAE, and R² for the trained model on the held-out
    test split. These three metrics together capture both absolute error
    magnitude (RMSE/MAE) and goodness-of-fit (R²).
    """
    y_pred = model.predict(X_test, verbose=0).flatten()
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred)) if len(y_test) > 1 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2}


def save_scaler(scaler, path: str = SCALER_PATH) -> None:
    """Persist the fitted StandardScaler alongside the model so inference
    can reproduce identical feature scaling."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)


def load_scaler(path: str = SCALER_PATH):
    """Load a previously saved StandardScaler. Returns None if not found."""
    if os.path.exists(path):
        return joblib.load(path)
    return None


def load_lstm_model(path: str = MODEL_PATH):
    """Load a previously trained LSTM model from disk. Returns None if the
    file doesn't exist, allowing the app to fall back to the plain
    Open-Meteo forecast when no trained model is available yet.

    If TensorFlow is not available, we try to load the fallback Scikit-Learn
    Random Forest Regressor.
    """
    try:
        if os.path.exists(path):
            from tensorflow.keras.models import load_model
            from tensorflow.keras.optimizers import Adam
            model = load_model(path, compile=False)
            model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
            return model
    except (ImportError, ModuleNotFoundError):
        pass

    fallback_path = os.path.join(MODEL_DIR, "fallback_model.joblib")
    if os.path.exists(fallback_path):
        return joblib.load(fallback_path)
    return None


def predict_next_temperature(model, scaler, sequence_df, feature_columns) -> float:
    """Run inference for a single lookback window.

    `sequence_df` must be a DataFrame with exactly `lookback` rows (most
    recent last) and the columns listed in `feature_columns`. Returns the
    predicted next-day mean temperature in Celsius.
    """
    scaled = scaler.transform(sequence_df[feature_columns])
    if 'keras' in str(type(model)):
        X = np.expand_dims(scaled, axis=0)  # add batch dimension
        pred = model.predict(X, verbose=0)
        return float(pred.flatten()[0])
    else:
        # Fallback model (Scikit-Learn Random Forest or similar)
        # Flatten the input sequence (lookback * num_features)
        X = scaled.reshape(1, -1)
        pred = model.predict(X)
        return float(pred[0])


def model_files_exist() -> bool:
    """Convenience check used by the Streamlit app to decide whether to
    show AI-prediction features or a 'train the model first' message.
    Checks for either LSTM model or fallback Random Forest model."""
    has_lstm = os.path.exists(MODEL_PATH)
    has_fallback = os.path.exists(os.path.join(MODEL_DIR, "fallback_model.joblib"))
    return (has_lstm or has_fallback) and os.path.exists(SCALER_PATH)

