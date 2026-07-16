"""
train_model.py
---------------
Standalone script to train the LSTM temperature-prediction model.

Run this once (or periodically) BEFORE launching app.py so the "AI Weather
Prediction" tab has a trained model available:

    python train_model.py --lat 28.6139 --lon 77.2090 --years 5

This will:
    1. Fetch `years` of daily historical weather for (lat, lon) from the
       Open-Meteo Archive API.
    2. Clean, feature-engineer, and scale the data (preprocess.py).
    3. Train an LSTM with early stopping + checkpointing (model.py).
    4. Save the trained model to models/lstm_weather_model.h5 and the
       fitted scaler to models/scaler.joblib.
    5. Print RMSE / MAE / R² on the held-out test split.
    6. Save the cleaned dataset to data/historical_weather.csv for reuse.
"""

import argparse
import sys
from datetime import date, timedelta

from utils import get_historical_weather
from preprocess import full_preprocessing_pipeline
from model import train_lstm, evaluate_model, save_scaler, MODEL_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="Train the weather prediction LSTM.")
    parser.add_argument("--lat", type=float, default=28.6139, help="Latitude of training location (default: New Delhi)")
    parser.add_argument("--lon", type=float, default=77.2090, help="Longitude of training location (default: New Delhi)")
    parser.add_argument("--years", type=int, default=5, help="Number of years of historical data to fetch")
    parser.add_argument("--lookback", type=int, default=14, help="Number of past days used to predict the next day")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum training epochs (early stopping may end sooner)")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    return parser.parse_args()


def main():
    args = parse_args()

    end_date = date.today() - timedelta(days=5)  # archive API has a short reporting lag
    start_date = end_date - timedelta(days=365 * args.years)

    print(f"Fetching {args.years} years of historical weather for "
          f"({args.lat}, {args.lon}) from {start_date} to {end_date} ...")
    historical_json = get_historical_weather(
        args.lat, args.lon, start_date.isoformat(), end_date.isoformat()
    )
    if historical_json is None:
        print("ERROR: Failed to fetch historical data. Check your internet connection.")
        sys.exit(1)

    print("Preprocessing data (cleaning, feature engineering, scaling, sequencing)...")
    pipeline_output = full_preprocessing_pipeline(
        historical_json, args.lat, args.lon, lookback=args.lookback
    )

    X_train, y_train = pipeline_output["X_train"], pipeline_output["y_train"]
    X_test, y_test = pipeline_output["X_test"], pipeline_output["y_test"]
    scaler = pipeline_output["scaler"]

    print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")
    if len(X_train) < 30:
        print("WARNING: Very little training data available. Consider increasing --years.")

    print("Training LSTM model (early stopping enabled, patience=10)...")
    try:
        model, history = train_lstm(
            X_train, y_train, X_test, y_test,
            epochs=args.epochs, batch_size=args.batch_size,
        )

        print("Evaluating on held-out test split...")
        metrics = evaluate_model(model, X_test, y_test)
        print(f"  RMSE : {metrics['rmse']:.3f} °C")
        print(f"  MAE  : {metrics['mae']:.3f} °C")
        print(f"  R²   : {metrics['r2']:.3f}")

        save_scaler(scaler)
        
        # Remove fallback model if it exists to keep directory clean
        import os
        fallback_path = os.path.join(MODEL_DIR, "fallback_model.joblib")
        if os.path.exists(fallback_path):
            os.remove(fallback_path)
    except (ImportError, ModuleNotFoundError):
        print("\nWARNING: TensorFlow is not installed or not supported in this environment.")
        print("Training a fallback Random Forest Regressor model instead...")
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        import joblib
        import os
        import numpy as np

        # Flatten sequences from (samples, lookback, features) to (samples, lookback * features)
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_flat, y_train)

        y_pred = model.predict(X_test_flat)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred)) if len(y_test) > 1 else float("nan")

        print("Evaluating fallback Random Forest on held-out test split...")
        print(f"  RMSE : {rmse:.3f} °C")
        print(f"  MAE  : {mae:.3f} °C")
        print(f"  R²   : {r2:.3f}")

        save_scaler(scaler)
        
        # Save Scikit-Learn fallback model
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(model, os.path.join(MODEL_DIR, "fallback_model.joblib"))
        
        # Clean up legacy LSTM model file if it exists to avoid mismatch
        lstm_path = os.path.join(MODEL_DIR, "lstm_weather_model.h5")
        if os.path.exists(lstm_path):
            os.remove(lstm_path)

    # Ensure data directory exists
    import os
    os.makedirs("data", exist_ok=True)
    pipeline_output["clean_df"].to_csv("data/historical_weather.csv", index=False)

    print(f"\nModel saved to {MODEL_DIR}/lstm_weather_model.h5")
    print(f"Scaler saved to {MODEL_DIR}/scaler.joblib")
    print("Cleaned dataset saved to data/historical_weather.csv")
    print("\nDone! You can now run: streamlit run app.py")


if __name__ == "__main__":
    main()
