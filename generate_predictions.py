import os
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from model_trainer import (
    load_symbols,
    get_stock_history_file,
    get_model_path,
    get_scaler_path,
    add_technical_indicators,
    get_stock_today_file,
)

DATA_DIR = "data"
PREDICTIONS_DIR = "predictions"

# Create predictions directory if it doesn't exist
os.makedirs(PREDICTIONS_DIR, exist_ok=True)


def get_next_trading_day(reference_date_str=None):
    """Calculate the next trading day (skip weekends)."""
    if reference_date_str:
        reference_date = datetime.strptime(reference_date_str, "%Y-%m-%d")
    else:
        reference_date = datetime.now()
    
    next_day = reference_date + timedelta(days=1)
    
    # Skip weekends
    while next_day.weekday() >= 5:  # Saturday=5, Sunday=6
        next_day += timedelta(days=1)
    
    return next_day


def load_history_data(symbol):
    """Load the full history CSV."""
    hist_file = get_stock_history_file(symbol)
    
    if not os.path.exists(hist_file):
        return None
    
    df = pd.read_csv(hist_file)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def get_last_day_data(symbol):
    """Get the most recent trading day's data from history (for graph)."""
    hist_file = get_stock_history_file(symbol)
    
    if not os.path.exists(hist_file):
        return None, None
    
    df = pd.read_csv(hist_file)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    if df.empty:
        return None, None
    
    # Get the last trading day's date
    last_date = df["timestamp"].dt.strftime("%Y-%m-%d").iloc[-1]
    
    # Filter to only that day's data
    last_day_df = df[df["timestamp"].dt.strftime("%Y-%m-%d") == last_date].copy()
    last_day_df = last_day_df.reset_index(drop=True)
    
    return last_day_df, last_date


def predict_next_hour(symbol, next_trading_day):
    """Predict next trading day's first hour prices at 1-min intervals (9:30 AM - 10:30 AM)."""
    try:
        model_path = get_model_path(symbol)
        scaler_path = get_scaler_path(symbol)
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            return None, f"Model or scaler not found for {symbol}"
        
        # Load model and scaler
        model = load_model(model_path, compile=False)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Load history data
        df = load_history_data(symbol)
        if df is None or len(df) < 10:
            return None, f"Insufficient history data for {symbol}"
        
        # Add features
        df = add_technical_indicators(df)
        
        # Drop non-numeric columns
        columns_to_drop = [col for col in ['timestamp', 'symbol'] if col in df.columns]
        if columns_to_drop:
            df = df.drop(columns_to_drop, axis=1)
        
        # Ensure target is last
        target_col = "Mid_Price"
        if target_col in df.columns:
            cols = df.columns.tolist()
            cols.remove(target_col)
            cols.append(target_col)
            df = df[cols]
        
        # Scale data
        df_scaled = scaler.transform(df.values)
        
        # Generate predictions for 60 1-min intervals (1 hour: 9:30 AM - 10:30 AM)
        window = 10
        predictions = []
        current_data = df_scaled[-window:].copy()
        
        # Start at 9:30 AM on the next trading day
        base_time = next_trading_day.replace(hour=9, minute=30, second=0, microsecond=0)
        
        for i in range(60):  # 60 minutes = 1 hour
            # Reshape for prediction
            x_pred = np.reshape(current_data, (1, window, df_scaled.shape[1]))
            
            # Predict
            pred_scaled = model.predict(x_pred, verbose=0)[0][0]
            
            # Inverse transform
            dummy = np.zeros((1, df_scaled.shape[1]))
            dummy[:, -1] = pred_scaled
            pred_price = scaler.inverse_transform(dummy)[0, -1]
            
            pred_time = base_time + timedelta(minutes=i)
            predictions.append({
                "time": pred_time.strftime("%H:%M"),
                "price": float(pred_price)
            })
            
            # Update current data for next prediction
            new_row = current_data[-1].copy()
            new_row[-1] = pred_scaled
            current_data = np.vstack([current_data[1:], new_row])
        
        return predictions, None
    
    except Exception as e:
        return None, f"Error predicting for {symbol}: {str(e)}"


def generate_stock_prediction(symbol):
    """Generate complete prediction data for a stock."""
    print(f"\nGenerating predictions for {symbol}...")
    
    # Get last day data for graph
    last_day_df, last_date = get_last_day_data(symbol)
    if last_day_df is None:
        print(f"  ❌ No data found for {symbol}")
        return None
    
    # Calculate next trading day
    next_trading_day = get_next_trading_day(last_date)
    
    # Get predictions
    predictions, pred_error = predict_next_hour(symbol, next_trading_day)
    
    if predictions is None:
        print(f"  ❌ Prediction failed: {pred_error}")
        return None
    
    # Prepare graph data from last day
    last_day_times = last_day_df["timestamp"].dt.strftime("%H:%M").tolist()
    last_day_prices = last_day_df["close"].tolist()
    
    # Get current price and prediction stats
    last_price = float(last_day_df["close"].iloc[-1])
    predicted_open = float(predictions[0]["price"])
    change = predicted_open - last_price
    change_pct = (change / last_price) * 100
    
    # Build prediction data
    stock_data = {
        "symbol": symbol,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "last_trading_day": last_date,
        "next_trading_day": next_trading_day.strftime("%Y-%m-%d"),
        "last_day_graph": {
            "times": last_day_times,
            "prices": last_day_prices,
            "title": f"{symbol} - Last Trading Day ({last_date})",
            "last_price": last_price
        },
        "predictions": {
            "times": [p["time"] for p in predictions],
            "prices": [p["price"] for p in predictions],
            "title": f"{symbol} - Next Day Prediction ({next_trading_day.strftime('%Y-%m-%d')}, 9:30-10:30 AM ET)",
            "current_price": last_price,
            "predicted_open": predicted_open,
            "change": change,
            "change_pct": change_pct
        }
    }
    
    print(f"  ✓ Generated predictions: {len(predictions)} data points")
    print(f"    Current Price: ${last_price:.2f}")
    print(f"    Predicted Open: ${predicted_open:.2f}")
    print(f"    Expected Change: ${change:.2f} ({change_pct:.2f}%)")
    
    return stock_data


def main():
    """Main prediction generation orchestration."""
    print("=" * 60)
    print("Stock Prediction Generator - Static JSON Output")
    print(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    symbols = load_symbols()
    print(f"\nGenerating predictions for {len(symbols)} stocks...\n")
    
    all_predictions = {}
    successful_count = 0
    failed_stocks = []
    
    for symbol in symbols:
        model_path = get_model_path(symbol)
        hist_file = get_stock_history_file(symbol)
        
        # Check if model and history exist
        if not os.path.exists(hist_file):
            print(f"  ❌ {symbol}: No history file found")
            failed_stocks.append(symbol)
            continue
        
        if not os.path.exists(model_path):
            print(f"  ❌ {symbol}: No model found")
            failed_stocks.append(symbol)
            continue
        
        # Generate predictions
        stock_data = generate_stock_prediction(symbol)
        
        if stock_data:
            all_predictions[symbol] = stock_data
            successful_count += 1
        else:
            failed_stocks.append(symbol)
    
    # Save all predictions to single JSON file
    output_file = os.path.join(PREDICTIONS_DIR, "all_predictions.json")
    with open(output_file, 'w') as f:
        json.dump(all_predictions, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Prediction generation completed!")
    print(f"  Successfully generated: {successful_count}/{len(symbols)} stocks")
    
    if failed_stocks:
        print(f"  Failed stocks: {', '.join(failed_stocks)}")
    
    print(f"  Output file: {output_file}")
    print("=" * 60)
    
    # Also save a summary file for quick access
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_stocks": len(symbols),
        "successful": successful_count,
        "failed": len(failed_stocks),
        "failed_stocks": failed_stocks,
        "stocks": list(all_predictions.keys())
    }
    
    summary_file = os.path.join(PREDICTIONS_DIR, "summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"  Summary file: {summary_file}")
    
    return successful_count == len(symbols)


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
