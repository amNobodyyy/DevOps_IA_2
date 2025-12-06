from flask import Flask, render_template, request, jsonify
import pandas as pd
import plotly.graph_objects as go
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from model_trainer import load_metrics_history, get_model_path, get_scaler_path, add_technical_indicators, get_stock_history_file
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pickle

app = Flask(__name__)

DATA_DIR = "data"

# Get available stocks from symbols.json
def get_available_stocks():
    """Load stock symbols from JSON file."""
    with open("symbols.json", "r") as f:
        config = json.load(f)
    return config.get("stocks", [])

def get_last_day_data(symbol):
    """Get the most recent trading day's data from the HISTORY file (for graph)."""
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


def load_history_data(symbol):
    """Load the full history CSV for predictions."""
    hist_file = get_stock_history_file(symbol)
    
    if not os.path.exists(hist_file):
        return None
    
    df = pd.read_csv(hist_file)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def get_next_trading_day(last_date_str):
    """Calculate the next trading day (skip weekends)."""
    last_date = datetime.strptime(last_date_str, "%Y-%m-%d")
    next_day = last_date + timedelta(days=1)
    
    # Skip weekends
    while next_day.weekday() >= 5:  # Saturday=5, Sunday=6
        next_day += timedelta(days=1)
    
    return next_day

def get_metrics_for_stock(symbol):
    """Get latest metrics for a stock."""
    metrics = load_metrics_history(symbol)
    if not metrics.get("history"):
        return None
    
    # Get latest training and prediction metrics
    latest_train = None
    latest_pred = None
    
    for entry in reversed(metrics["history"]):
        if entry["operation"] in ["train", "retrain"] and latest_train is None:
            latest_train = entry
        if entry["operation"] == "prediction" and latest_pred is None:
            latest_pred = entry
    
    return {
        "training": latest_train,
        "prediction": latest_pred
    }

def predict_next_hour(symbol, next_trading_day):
    """Predict next trading day's first hour prices at 1-min intervals (9:30 AM - 10:30 AM).
    
    Uses the HISTORY CSV for predictions (trained model expects full history context).
    """
    try:
        model_path = get_model_path(symbol)
        scaler_path = get_scaler_path(symbol)
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            return None, "Model or scaler not found"
        
        # Load model and scaler (trained by model_trainer.py)
        model = load_model(model_path, compile=False)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Load HISTORY data for predictions (includes all appended daily data)
        df = load_history_data(symbol)
        if df is None or len(df) < 10:
            return None, "Insufficient history data for prediction"
        
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
        
        # Scale data using the trained scaler
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
                "full_time": pred_time.strftime("%Y-%m-%d %H:%M"),
                "price": float(pred_price)
            })
            
            # Update current data for next prediction
            new_row = current_data[-1].copy()
            new_row[-1] = pred_scaled
            current_data = np.vstack([current_data[1:], new_row])
        
        return predictions, None
    
    except Exception as e:
        return None, str(e)

@app.route("/")
def home():
    stocks = get_available_stocks()
    return render_template("index.html", stocks=stocks)

@app.route("/api/symbols", methods=["GET"])
def get_symbols():
    """API endpoint to get available stock symbols."""
    try:
        stocks = get_available_stocks()
        return jsonify({"symbols": stocks}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/analyze", methods=["POST"])
def analyze():
    """API endpoint for stock analysis."""
    try:
        data = request.get_json()
        symbol = data.get("symbol")
        
        if not symbol:
            return jsonify({"error": "Symbol required"}), 400
        
        # Get last trading day's data
        last_day, last_date_str = get_last_day_data(symbol)
        if last_day is None:
            return jsonify({"error": "No data found for symbol"}), 404
        
        # Calculate next trading day for predictions
        next_trading_day = get_next_trading_day(last_date_str)
        
        # Create last day graph (9:30 AM - 4:00 PM market hours)
        fig_last = go.Figure()
        fig_last.add_trace(go.Scatter(
            x=last_day["timestamp"].dt.strftime("%H:%M").tolist(),
            y=last_day["close"].tolist(),
            mode='lines+markers',
            name='Close Price',
            line=dict(color='blue', width=2),
            marker=dict(size=4)
        ))
        
        fig_last.update_layout(
            title=f"{symbol} - Last Trading Day ({last_date_str})",
            xaxis_title="Time (Market Hours: 9:30 AM - 4:00 PM ET)",
            yaxis_title="Price (USD)",
            hovermode='x unified',
            template='plotly_white',
            height=400
        )
        
        graph_last_day = fig_last.to_json()
        
        # Get predictions for next trading day's first hour (9:30 AM - 10:30 AM)
        # Uses HISTORY data for predictions (not last_day)
        predictions, pred_error = predict_next_hour(symbol, next_trading_day)
        
        graph_next_hour = None
        prediction_data = None
        
        if predictions:
            times = [p["time"] for p in predictions]
            prices = [p["price"] for p in predictions]
            
            fig_next = go.Figure()
            fig_next.add_trace(go.Scatter(
                x=times,
                y=prices,
                mode='lines+markers',
                name='Predicted Price',
                line=dict(color='green', width=2),
                marker=dict(size=3)
            ))
            
            # Add last closing price as reference
            last_price = last_day["close"].iloc[-1]
            fig_next.add_hline(y=last_price, line_dash="dash", line_color="red", 
                             annotation_text=f"Last Close: ${last_price:.2f}", annotation_position="right")
            
            next_date_str = next_trading_day.strftime("%Y-%m-%d")
            fig_next.update_layout(
                title=f"{symbol} - Next Day Prediction ({next_date_str}, 9:30-10:30 AM ET)",
                xaxis_title="Time (1-minute intervals)",
                yaxis_title="Predicted Price (USD)",
                hovermode='x unified',
                template='plotly_white',
                height=400
            )
            
            graph_next_hour = fig_next.to_json()
            prediction_data = {
                "predictions": predictions,
                "current_price": float(last_price),
                "predicted_open": float(predictions[0]["price"]),
                "change": float(predictions[0]["price"] - last_price),
                "last_date": last_date_str,
                "prediction_date": next_date_str
            }
        
        # Get metrics
        metrics = get_metrics_for_stock(symbol)
        
        metrics_display = None
        if metrics:
            if metrics["training"]:
                train = metrics["training"]
                # training metrics are nested under 'training' key in history entries
                trn = train.get("training", {}) if isinstance(train, dict) else {}
                metrics_display = {
                    "training": {
                        "r2_score": f"{trn.get('test_r2', 0):.4f}",
                        "rmse": f"{trn.get('test_rmse', 0):.6f}",
                        "mae": f"{trn.get('test_mae', 0):.6f}",
                        "train_r2": f"{trn.get('train_r2', 0):.4f}",
                        "timestamp": train.get("timestamp", "N/A")
                    }
                }
            
            if metrics["prediction"]:
                pred = metrics["prediction"]
                if metrics_display is None:
                    metrics_display = {}
                metrics_display["prediction"] = {
                    "last_close": f"${pred.get('last_actual_close', 0):.2f}",
                    "predicted_price": f"${pred.get('predicted_next_price', 0):.2f}",
                    "change": f"${pred.get('price_change', 0):.2f}",
                    "change_pct": f"{pred.get('price_change_percent', 0):.2f}%",
                    "timestamp": pred.get("timestamp", "N/A")
                }
        
        return jsonify({
            "success": True,
            "symbol": symbol,
            "graph_last_day": graph_last_day,
            "graph_next_hour": graph_next_hour,
            "predictions": prediction_data,
            "metrics": metrics_display,
            "last_date": last_date_str,
            "prediction_date": next_trading_day.strftime("%Y-%m-%d"),
            "error": pred_error
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Use PORT env var for platforms like Cloud Run; default to 8080
    app.run(host="0.0.0.0", port=4879, debug=True)