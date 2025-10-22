import os
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

DATA_DIR = "data"
MODELS_DIR = "models"
SYMBOLS_FILE = "symbols.json"

# Create models directory if it doesn't exist
os.makedirs(MODELS_DIR, exist_ok=True)


def load_symbols():
    """Load stock symbols from JSON file."""
    with open(SYMBOLS_FILE, "r") as f:
        config = json.load(f)
    return config.get("stocks", [])


def get_stock_history_file(symbol):
    """Get the path to historical data file for a stock."""
    return os.path.join(DATA_DIR, f"{symbol}_history.csv")


def get_stock_today_file(symbol):
    """Get the path to today's data file for a stock."""
    today_str = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(DATA_DIR, f"{symbol}_{today_str}.csv")


def get_model_path(symbol):
    """Get the path to model file for a stock."""
    return os.path.join(MODELS_DIR, f"{symbol}_model.pkl")


def get_scaler_path(symbol):
    """Get the path to scaler file for a stock."""
    return os.path.join(MODELS_DIR, f"{symbol}_scaler.pkl")


def load_stock_data(symbol, include_today=False):
    """
    Load stock data for a symbol.
    If include_today=True, appends today's data to historical data.
    """
    hist_file = get_stock_history_file(symbol)
    
    if not os.path.exists(hist_file):
        print(f"ERROR: History file not found for {symbol}: {hist_file}")
        return None

    df = pd.read_csv(hist_file)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Append today's data if available and requested
    if include_today:
        today_file = get_stock_today_file(symbol)
        if os.path.exists(today_file):
            today_df = pd.read_csv(today_file)
            today_df["timestamp"] = pd.to_datetime(today_df["timestamp"])
            df = pd.concat([df, today_df], ignore_index=True)
            print(f"  Appended today's {len(today_df)} records")

    # Remove duplicates and sort
    df = df.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"  Loaded {len(df)} records for {symbol}")
    return df


def create_features(df, lookback=5):
    """
    Create features for the model.
    Lookback window: uses last 'lookback' days to predict next close.
    """
    df = df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Feature engineering
    df["close_lag1"] = df["close"].shift(1)
    df["close_lag2"] = df["close"].shift(2)
    df["volume_lag1"] = df["volume"].shift(1)
    df["high_low_ratio"] = df["high"] / df["low"]
    df["close_open_ratio"] = df["close"] / df["open"]

    # Rolling features
    df["close_sma5"] = df["close"].rolling(window=5, min_periods=1).mean()
    df["volume_sma5"] = df["volume"].rolling(window=5, min_periods=1).mean()

    # Drop rows with NaN values created by lag/rolling
    df = df.dropna().reset_index(drop=True)

    return df


def prepare_training_data(df):
    """Prepare X (features) and y (target) for training."""
    feature_cols = [
        "open", "high", "low", "close_lag1", "close_lag2",
        "volume", "volume_lag1", "high_low_ratio", "close_open_ratio",
        "close_sma5", "volume_sma5"
    ]

    X = df[feature_cols].values
    y = df["close"].values

    return X, y, feature_cols


def train_stock_model(symbol, retrain=False):
    """
    Train a new model for a stock or retrain existing one.
    retrain=True: Use existing model as checkpoint, fine-tune with new data
    retrain=False: Train new model from scratch
    """
    print(f"\nTraining model for {symbol}...")

    # Load data
    df = load_stock_data(symbol, include_today=True)
    if df is None or len(df) < 20:
        print(f"ERROR: Insufficient data for {symbol} (need at least 20 records)")
        return False

    # Create features
    df = create_features(df, lookback=5)
    print(f"  Created features ({len(df)} samples)")

    # Prepare training data
    X, y, feature_cols = prepare_training_data(df)

    # Normalize features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Load or create model
    model_path = get_model_path(symbol)
    scaler_path = get_scaler_path(symbol)

    if retrain and os.path.exists(model_path):
        print(f"  Loading existing model for fine-tuning...")
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        # Re-scale with new data
        X_scaled = scaler.fit_transform(X)
    else:
        print(f"  Creating new model...")
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

    # Train/update model
    model.fit(X_scaled, y)
    print(f"  Model training completed")

    # Evaluate on training data
    y_pred = model.predict(X_scaled)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print(f"  Training Metrics:")
    print(f"    MSE: {mse:.4f}")
    print(f"    MAE: {mae:.4f}")
    print(f"    R2: {r2:.4f}")

    # Save model and scaler
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"  Model saved: {model_path}")
    print(f"  Scaler saved: {scaler_path}")

    return True


def update_stock_model(symbol):
    """
    Update existing model with new data.
    Model is fine-tuned, not completely retrained.
    """
    print(f"\nUpdating model for {symbol}...")

    model_path = get_model_path(symbol)
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found for {symbol}. Train first with train=True")
        return False

    # This calls train with retrain=True
    return train_stock_model(symbol, retrain=True)


def predict_next_close(symbol):
    """Predict the next close price for a stock."""
    print(f"\nPredicting next close for {symbol}...")

    model_path = get_model_path(symbol)
    scaler_path = get_scaler_path(symbol)

    if not os.path.exists(model_path):
        print(f"ERROR: Model not found for {symbol}")
        return None

    # Load model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Load latest data
    df = load_stock_data(symbol, include_today=True)
    if df is None or len(df) < 20:
        print(f"ERROR: Insufficient data for prediction")
        return None

    # Create features
    df = create_features(df, lookback=5)

    # Get last row as features
    feature_cols = [
        "open", "high", "low", "close_lag1", "close_lag2",
        "volume", "volume_lag1", "high_low_ratio", "close_open_ratio",
        "close_sma5", "volume_sma5"
    ]
    
    last_row = df[feature_cols].values[-1].reshape(1, -1)
    last_row_scaled = scaler.transform(last_row)

    # Predict
    predicted_close = model.predict(last_row_scaled)[0]
    last_actual_close = df["close"].values[-1]

    print(f"  Last actual close: ${last_actual_close:.2f}")
    print(f"  Predicted next close: ${predicted_close:.2f}")
    print(f"  Change: ${predicted_close - last_actual_close:.2f}")

    return predicted_close


def main():
    """Main model training orchestration."""
    print("=" * 60)
    print("Stock Model Trainer - MLOps Pipeline")
    print("=" * 60)

    symbols = load_symbols()
    print(f"\nTraining models for {len(symbols)} stocks...")

    trained_count = 0
    updated_count = 0

    for symbol in symbols:
        model_path = get_model_path(symbol)
        hist_file = get_stock_history_file(symbol)

        # Check if stock data exists
        if not os.path.exists(hist_file):
            print(f"\nSkipping {symbol} - no historical data found")
            continue

        # Check if model exists
        if os.path.exists(model_path):
            # Update existing model
            success = update_stock_model(symbol)
            if success:
                updated_count += 1
        else:
            # Train new model
            success = train_stock_model(symbol, retrain=False)
            if success:
                trained_count += 1

    # Make predictions for all models
    print("\n" + "=" * 60)
    print("Making predictions...")
    print("=" * 60)

    for symbol in symbols:
        model_path = get_model_path(symbol)
        if os.path.exists(model_path):
            predict_next_close(symbol)

    print("\n" + "=" * 60)
    print("Model training completed!")
    print(f"  New models trained: {trained_count}")
    print(f"  Existing models updated: {updated_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
