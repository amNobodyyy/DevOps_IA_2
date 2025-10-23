from flask import Flask, render_template, request, jsonify
import pandas as pd
import plotly.graph_objects as go
import json
import os
from datetime import datetime, timedelta
import numpy as np
from model_trainer import load_metrics_history, get_model_path, get_scaler_path, add_technical_indicators, load_stock_data
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pickle

app = Flask(__name__)

# Get available stocks from symbols.json
def get_available_stocks():
    """Load stock symbols from JSON file."""
    with open("symbols.json", "r") as f:
        config = json.load(f)
    return config.get("stocks", [])

def get_last_day_data(symbol):
    """Get yesterday's data for the stock."""
    today_str = datetime.now().strftime("%Y-%m-%d")
    today_file = os.path.join("data", f"{symbol}_{today_str}.csv")
    
    if os.path.exists(today_file):
        df = pd.read_csv(today_file)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df.sort_values("timestamp").reset_index(drop=True)
    
    return None

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

def predict_next_hour(symbol):
    """Predict next hour's prices at 5-min intervals."""
    try:
        model_path = get_model_path(symbol)
        scaler_path = get_scaler_path(symbol)
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            return None, "Model or scaler not found"
        
        # Load model and scaler
        model = load_model(model_path, compile=False)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Load data
        df = load_stock_data(symbol, include_today=True)
        if df is None or len(df) < 50:
            return None, "Insufficient data"
        
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
        
        # Generate predictions for next 12 5-min intervals (1 hour)
        window = 10
        predictions = []
        current_data = df_scaled[-window:].copy()
        
        base_time = datetime.now() + timedelta(minutes=5)
        
        for i in range(12):
            # Reshape for prediction
            x_pred = np.reshape(current_data, (1, window, df_scaled.shape[1]))
            
            # Predict
            pred_scaled = model.predict(x_pred, verbose=0)[0][0]
            
            # Inverse transform
            dummy = np.zeros((1, df_scaled.shape[1]))
            dummy[:, -1] = pred_scaled
            pred_price = scaler.inverse_transform(dummy)[0, -1]
            
            pred_time = base_time + timedelta(minutes=i*5)
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
        return None, str(e)

@app.route("/")
def home():
    stocks = get_available_stocks()
    return render_template("index.html", stocks=stocks)

@app.route("/api/analyze", methods=["POST"])
def analyze():
    """API endpoint for stock analysis."""
    try:
        data = request.get_json()
        symbol = data.get("symbol")
        
        if not symbol:
            return jsonify({"error": "Symbol required"}), 400
        
        # Get last day's data
        last_day = get_last_day_data(symbol)
        if last_day is None:
            return jsonify({"error": "No data found for symbol"}), 404
        
        # Create last day graph
        fig_last = go.Figure()
        fig_last.add_trace(go.Scatter(
            x=last_day["timestamp"].dt.strftime("%H:%M").tolist(),
            y=last_day["close"].tolist(),
            mode='lines+markers',
            name='Close Price',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ))
        
        fig_last.update_layout(
            title=f"{symbol} - Last Trading Day",
            xaxis_title="Time",
            yaxis_title="Price (USD)",
            hovermode='x unified',
            template='plotly_white',
            height=400
        )
        
        graph_last_day = fig_last.to_json()
        
        # Get predictions for next hour
        predictions, pred_error = predict_next_hour(symbol)
        
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
                marker=dict(size=6)
            ))
            
            # Add current price as reference
            last_price = last_day["close"].iloc[-1]
            fig_next.add_hline(y=last_price, line_dash="dash", line_color="red", 
                             annotation_text="Current Price", annotation_position="right")
            
            fig_next.update_layout(
                title=f"{symbol} - Next Hour Prediction (5-min intervals)",
                xaxis_title="Time",
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
                "change": float(predictions[0]["price"] - last_price)
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
            "error": pred_error
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Use PORT env var for platforms like Cloud Run; default to 8080
    port = int(os.environ.get("PORT", 8080))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)