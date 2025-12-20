import os
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import time

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt

from tensorflow.keras.layers import Dense, Activation, Dropout, Bidirectional, LSTM
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')

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
    """Get the path to the most recent daily data file for a stock."""
    # Look for the most recent daily file (not history)
    # Files are named: {SYMBOL}_{DATE}.csv
    data_dir = Path(DATA_DIR)
    pattern = f"{symbol}_????-??-??.csv"
    
    daily_files = list(data_dir.glob(pattern))
    if not daily_files:
        return None
    
    # Sort by filename (date) and get the most recent
    daily_files.sort(reverse=True)
    return str(daily_files[0])


def get_model_path(symbol):
    """Get the path to model file for a stock."""
    return os.path.join(MODELS_DIR, f"{symbol}_model.h5")


def get_scaler_path(symbol):
    """Get the path to scaler file for a stock."""
    return os.path.join(MODELS_DIR, f"{symbol}_scaler.pkl")


def get_metrics_path(symbol):
    """Get the path to metrics file for a stock."""
    return os.path.join(MODELS_DIR, f"{symbol}_metrics.json")


def load_metrics_history(symbol):
    """Load historical metrics for a stock."""
    metrics_path = get_metrics_path(symbol)
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return {"symbol": symbol, "history": []}


def save_metrics(symbol, metrics_data):
    """Save metrics incrementally to JSON file."""
    metrics_path = get_metrics_path(symbol)
    
    # Load existing metrics
    metrics_history = load_metrics_history(symbol)
    
    # Add timestamp if not present
    if "timestamp" not in metrics_data:
        metrics_data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Add date for easy filtering
    if "date" not in metrics_data:
        metrics_data["date"] = datetime.now().strftime("%Y-%m-%d")
    
    # Append new metrics to history
    metrics_history["history"].append(metrics_data)
    
    # Save updated metrics
    with open(metrics_path, 'w') as f:
        json.dump(metrics_history, f, indent=2)
    
    print(f"  Metrics saved: {metrics_path}")


def load_stock_data(symbol, include_today=False):
    """
    Load stock data for a symbol from history file.
    Does NOT append daily data - that's done separately in append_daily_to_history().
    """
    hist_file = get_stock_history_file(symbol)
    
    if not os.path.exists(hist_file):
        print(f"ERROR: History file not found for {symbol}: {hist_file}")
        return None

    df = pd.read_csv(hist_file)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Remove duplicates and sort
    df = df.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"  Loaded {len(df)} records for {symbol}")
    return df


def append_daily_to_history(symbol):
    """
    Append the most recent daily data file to the history CSV.
    This permanently adds the new data to history.
    Returns True if data was appended, False otherwise.
    """
    hist_file = get_stock_history_file(symbol)
    today_file = get_stock_today_file(symbol)
    
    if not today_file or not os.path.exists(today_file):
        print(f"  No daily data file found for {symbol}")
        return False
    
    if not os.path.exists(hist_file):
        print(f"  No history file found for {symbol}")
        return False
    
    # Load history
    hist_df = pd.read_csv(hist_file)
    hist_df["timestamp"] = pd.to_datetime(hist_df["timestamp"])
    
    # Load daily data
    daily_df = pd.read_csv(today_file)
    daily_df["timestamp"] = pd.to_datetime(daily_df["timestamp"])
    
    # Check if this daily data is already in history (avoid duplicates)
    daily_date = daily_df["timestamp"].iloc[0].strftime("%Y-%m-%d")
    existing_dates = hist_df["timestamp"].dt.strftime("%Y-%m-%d").unique()
    
    if daily_date in existing_dates:
        print(f"  Daily data for {daily_date} already in history, skipping append")
        return False
    
    # Append daily data to history
    combined_df = pd.concat([hist_df, daily_df], ignore_index=True)
    
    # Remove duplicates and sort
    combined_df = combined_df.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    combined_df = combined_df.sort_values("timestamp").reset_index(drop=True)
    
    # Save updated history
    combined_df.to_csv(hist_file, index=False)
    print(f"  ✓ Appended {len(daily_df)} records from {daily_date} to history")
    print(f"  ✓ Updated history saved: {hist_file} (total: {len(combined_df)} records)")
    
    # Delete the daily file after successful append
    try:
        os.remove(today_file)
        print(f"  ✓ Deleted daily file: {today_file}")
    except Exception as e:
        print(f"  ⚠️ Warning: Could not delete daily file {today_file}: {e}")
    
    return True


def add_technical_indicators(df):
    """Add technical indicators to the dataframe."""
    try:
        # Simple Moving Averages
        df['SMA_5'] = df['close'].rolling(window=5).mean()
        df['SMA_10'] = df['close'].rolling(window=10).mean()
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        
        # Price momentum
        df['Price_Change'] = df['close'].pct_change()
        df['Price_Change_2'] = df['close'].pct_change(2)
        
        # Volatility
        df['Volatility'] = df['close'].rolling(window=10).std()
        
        # High-Low ratio
        df['HL_Ratio'] = df['high'] / df['low']
        
        # Volume indicators
        if 'volume' in df.columns:
            df['Volume_MA'] = df['volume'].rolling(window=5).mean()
            df['Volume_Ratio'] = df['volume'] / (df['Volume_MA'] + 1e-10)
        
        # Mid price
        df['Mid_Price'] = (df['high'] + df['low']) / 2.0
        
        # Forward fill and backward fill NaN values
        df = df.ffill()
        df = df.bfill()
        df.fillna(0, inplace=True)
        
        return df
    except Exception as e:
        print(f"  Error adding technical indicators: {e}")
        return df


def create_features(df, lookback=5):
    """
    Create features for the model - REMOVED, now using technical indicators.
    """
    # Just add technical indicators and return
    df = add_technical_indicators(df)
    return df


def prepare_sequences(df, window=10):
    """Prepare sequences for LSTM training."""
    # Drop non-numeric columns
    columns_to_drop = []
    if 'timestamp' in df.columns:
        columns_to_drop.append('timestamp')
    if 'symbol' in df.columns:
        columns_to_drop.append('symbol')
    
    if columns_to_drop:
        df = df.drop(columns_to_drop, axis=1)
    
    # Ensure target column is last
    target_col = "Mid_Price"
    if target_col in df.columns:
        cols = df.columns.tolist()
        cols.remove(target_col)
        cols.append(target_col)
        df = df[cols]
    
    # Scale the data
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df.values)
    
    # Prepare sequences
    sequences = []
    for i in range(len(df_scaled) - window):
        sequences.append(df_scaled[i:i + window + 1])
    
    if len(sequences) == 0:
        return None, None, None, None, None
    
    sequences = np.array(sequences)
    
    # Split data (80/20)
    split_ratio = 0.8
    train_size = int(len(sequences) * split_ratio)
    
    train_sequences = sequences[:train_size]
    test_sequences = sequences[train_size:]
    
    if len(train_sequences) == 0 or len(test_sequences) == 0:
        return None, None, None, None, None
    
    # Prepare training data
    X_train = train_sequences[:, :-1]
    y_train = train_sequences[:, -1][:, -1]  # Last feature is target
    X_test = test_sequences[:, :-1]
    y_test = test_sequences[:, -1][:, -1]
    
    return X_train, y_train, X_test, y_test, scaler


def build_lstm_model(input_shape, learning_rate=0.001, bidirectional=False):
    """Build an LSTM model for stock price prediction."""
    model = Sequential()
    
    if bidirectional:
        model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape))
    else:
        model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    
    model.add(Dropout(0.3))
    
    if bidirectional:
        model.add(Bidirectional(LSTM(128, return_sequences=False)))
    else:
        model.add(LSTM(128, return_sequences=False))
    
    model.add(Dropout(0.3))
    model.add(Dense(1))
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model


def train_stock_model(symbol, retrain=False):
    """
    Train a new LSTM model for a stock or update existing one.
    retrain=True: Load existing model and continue training
    retrain=False: Train new model from scratch
    
    Trains on the FULL history data (after daily data has been appended).
    """
    print(f"\nTraining LSTM model for {symbol}...")

    # Load data from history file (already includes appended daily data)
    df = load_stock_data(symbol)
    if df is None or len(df) < 50:
        print(f"ERROR: Insufficient data for {symbol} (need at least 50 records)")
        return False

    # Add technical indicators
    df = create_features(df)
    print(f"  Added technical indicators ({len(df)} samples)")

    # Prepare sequences for LSTM
    window = 10
    X_train, y_train, X_test, y_test, scaler = prepare_sequences(df, window=window)
    
    if X_train is None:
        print(f"ERROR: Could not create sequences for {symbol}")
        return False
    
    print(f"  Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Model configuration
    model_path = get_model_path(symbol)
    scaler_path = get_scaler_path(symbol)
    
    config = {
        'lstm_units_1': 64,
        'lstm_units_2': 128,
        'dropout_rate': 0.3,
        'learning_rate': 0.001,
        'bidirectional': False
    }

    # Load or create model
    if retrain and os.path.exists(model_path):
        print(f"  Loading existing model for continued training...")
        model = load_model(model_path, compile=False)
        model.compile(optimizer=Adam(learning_rate=config['learning_rate']), loss='mse', metrics=['mae'])
        # Load existing scaler
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    else:
        print(f"  Creating new LSTM model...")
        amount_of_features = X_train.shape[2]
        model = build_lstm_model(
            (window, amount_of_features), 
            learning_rate=config['learning_rate'],
            bidirectional=config['bidirectional']
        )

    # Enhanced training with callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=0.0001, verbose=0)
    ]

    # Train model
    print("  Training LSTM model...")
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=50,
        validation_split=0.15,
        callbacks=callbacks,
        verbose=0
    )

    # Evaluate model
    train_loss = model.evaluate(X_train, y_train, verbose=0)
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    
    # Make predictions for R² calculation
    y_train_pred = model.predict(X_train, verbose=0)
    y_test_pred = model.predict(X_test, verbose=0)
    
    # Calculate additional metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"  Model training completed")
    print(f"  Training Metrics:")
    print(f"    Train Loss (MSE): {train_loss[0]:.6f}")
    print(f"    Test Loss (MSE): {test_loss[0]:.6f}")
    print(f"    Test RMSE: {sqrt(test_loss[0]):.6f}")
    print(f"    Test R²: {test_r2:.6f}")

    # Prepare metrics data
    metrics_data = {
        "operation": "retrain" if retrain else "train",
        "samples": {
            "train": len(X_train),
            "test": len(X_test),
            "total": len(df)
        },
        "training": {
            "train_loss_mse": float(train_loss[0]),
            "train_mae": float(train_mae),
            "train_r2": float(train_r2),
            "test_loss_mse": float(test_loss[0]),
            "test_mae": float(test_mae),
            "test_rmse": float(sqrt(test_loss[0])),
            "test_r2": float(test_r2)
        },
        "model_config": config,
        "window_size": window,
        "epochs_run": len(history.history['loss']),
        "final_train_loss": float(history.history['loss'][-1]),
        "final_val_loss": float(history.history['val_loss'][-1])
    }
    
    # Save metrics incrementally
    save_metrics(symbol, metrics_data)

    # Save model and scaler
    model.save(model_path)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
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
    """Predict the next price for a stock using LSTM model."""
    print(f"\nPredicting next price for {symbol}...")

    model_path = get_model_path(symbol)
    scaler_path = get_scaler_path(symbol)

    if not os.path.exists(model_path):
        print(f"ERROR: Model not found for {symbol}")
        return None

    # Load model and scaler (compile=False to avoid deserialization issues)
    model = load_model(model_path, compile=False)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # Load history data (already includes all appended daily data)
    df = load_stock_data(symbol)
    if df is None or len(df) < 50:
        print(f"ERROR: Insufficient data for prediction")
        return None

    # Add technical indicators
    df = create_features(df)
    
    # Prepare data for prediction - drop non-numeric columns
    columns_to_drop = []
    if 'timestamp' in df.columns:
        columns_to_drop.append('timestamp')
    if 'symbol' in df.columns:
        columns_to_drop.append('symbol')
    
    if columns_to_drop:
        df = df.drop(columns_to_drop, axis=1)
    
    # Ensure target column is last
    target_col = "Mid_Price"
    if target_col in df.columns:
        cols = df.columns.tolist()
        cols.remove(target_col)
        cols.append(target_col)
        df = df[cols]
    
    # Scale the data
    df_scaled = scaler.transform(df.values)
    
    # Get last window for prediction
    window = 10
    if len(df_scaled) < window:
        print(f"ERROR: Not enough data for window size {window}")
        return None
    
    latest_data = df_scaled[-window:]
    x_latest = np.reshape(latest_data, (1, window, df_scaled.shape[1]))
    
    # Predict
    p_latest_scaled = model.predict(x_latest, verbose=0)
    
    # Inverse transform prediction
    amount_of_features = df_scaled.shape[1]
    dummy_latest = np.zeros((1, amount_of_features))
    dummy_latest[:, -1] = p_latest_scaled.flatten()
    predicted_price = scaler.inverse_transform(dummy_latest)[0, -1]
    
    # Get last actual price
    last_actual_close = df["close"].values[-1] if "close" in df.columns else df[target_col].values[-1]

    print(f"  Last actual close: ${last_actual_close:.2f}")
    print(f"  Predicted next price: ${predicted_price:.2f}")
    print(f"  Change: ${predicted_price - last_actual_close:.2f}")

    # Calculate prediction metrics
    price_change = predicted_price - last_actual_close
    price_change_pct = (price_change / last_actual_close) * 100
    
    # Prepare prediction metrics
    prediction_metrics = {
        "operation": "prediction",
        "last_actual_close": float(last_actual_close),
        "predicted_next_price": float(predicted_price),
        "price_change": float(price_change),
        "price_change_percent": float(price_change_pct),
        "data_points_used": len(df),
        "window_size": window
    }
    
    # Save prediction metrics
    save_metrics(symbol, prediction_metrics)

    return predicted_price


def main():
    """Main model training orchestration.
    
    Flow:
    1. For each existing stock: Append daily data to history CSV
    2. Train/retrain model on full history
    3. Make predictions
    """
    print("=" * 60)
    print("Stock Model Trainer - MLOps Pipeline")
    print("=" * 60)

    symbols = load_symbols()
    print(f"\nProcessing {len(symbols)} stocks...")

    trained_count = 0
    updated_count = 0
    appended_count = 0

    for symbol in symbols:
        model_path = get_model_path(symbol)
        hist_file = get_stock_history_file(symbol)

        # Check if stock data exists
        if not os.path.exists(hist_file):
            print(f"\nSkipping {symbol} - no historical data found")
            continue

        print(f"\n{'='*40}")
        print(f"Processing: {symbol}")
        print(f"{'='*40}")

        # Step 1: Append daily data to history (if available)
        if append_daily_to_history(symbol):
            appended_count += 1

        # Step 2: Train or retrain model on full history
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

    # Step 3: Make predictions for all models
    print("\n" + "=" * 60)
    print("Making predictions...")
    print("=" * 60)

    for symbol in symbols:
        model_path = get_model_path(symbol)
        if os.path.exists(model_path):
            predict_next_close(symbol)

    print("\n" + "=" * 60)
    print("Model training completed!")
    print(f"  Daily data appended: {appended_count} stocks")
    print(f"  New models trained: {trained_count}")
    print(f"  Existing models updated: {updated_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
