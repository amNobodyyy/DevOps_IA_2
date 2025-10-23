from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle
from tensorflow.keras.models import load_model
import os
import re
import json
from datetime import datetime
from model_trainer import load_stock_data, create_features, get_model_path, get_scaler_path

app = Flask(__name__)

# ======== CONFIG ========
DATA_DIR = "data/"
MODEL_DIR = "models/"
WINDOW_SIZE = 10  # must match model_trainer.py

# ======== Helper: Get latest CSV for each stock ========
def get_latest_stock_files():
    files = [f for f in os.listdir(DATA_DIR) if re.match(r"^[A-Z]+_\d{4}-\d{2}-\d{2}\.csv$", f)]
    stock_map = {}

    for f in files:
        match = re.match(r"^([A-Z]+)_(\d{4}-\d{2}-\d{2})\.csv$", f)
        if match:
            symbol, date_str = match.groups()
            date = datetime.strptime(date_str, "%Y-%m-%d").date()
            if symbol not in stock_map or date > stock_map[symbol]["date"]:
                stock_map[symbol] = {"file": f, "date": date}
    return stock_map

# Discover available stock files dynamically
STOCK_FILES = get_latest_stock_files()
STOCKS = list(STOCK_FILES.keys())

@app.route("/", methods=["GET", "POST"])
def home():
    selected_stock = None
    graphJSON = None
    prediction = None
    last_close = None

    if request.method == "POST":
        selected_stock = request.form.get("stock")
        
        # Load stock data including today's intraday
        stock_data_original = load_stock_data(selected_stock, include_today=True)
        
        if stock_data_original is None or len(stock_data_original) < WINDOW_SIZE:
            prediction = "⚠️ Not enough data for prediction"
        else:
            # Find the latest intraday CSV file for this stock
            stock_files = [f for f in os.listdir(DATA_DIR) if f.startswith(f"{selected_stock}_") and f.endswith(".csv")]
            
            if stock_files:
                # Extract dates and find the latest
                dates = []
                for f in stock_files:
                    match = re.match(rf"^{selected_stock}_(\d{{4}}-\d{{2}}-\d{{2}})\.csv$", f)
                    if match:
                        date_str = match.group(1)
                        dates.append((date_str, f))
                
                if dates:
                    # Sort by date and get the latest
                    dates.sort(reverse=True)
                    latest_date_str, latest_file = dates[0]
                    latest_date = datetime.strptime(latest_date_str, "%Y-%m-%d").date()
                    
                    # Load latest intraday file
                    latest_intraday = pd.read_csv(os.path.join(DATA_DIR, latest_file))
                    latest_intraday["timestamp"] = pd.to_datetime(latest_intraday["timestamp"])
                    
                    # Store latest intraday data for plotting
                    plot_timestamps = latest_intraday["timestamp"].tolist()
                    plot_closes = latest_intraday["close"].tolist()
                    plot_date = latest_date_str
                else:
                    plot_timestamps = []
                    plot_closes = []
                    plot_date = None
            else:
                plot_timestamps = []
                plot_closes = []
                plot_date = None
            
            # Use ALL data (historical + today) for prediction
            last_close = stock_data_original["close"].iloc[-1]
            
            # Create a copy for model processing
            stock_data = stock_data_original.copy()
            
            # Add technical indicators
            stock_data = create_features(stock_data)

            # Drop non-numeric columns
            columns_to_drop = []
            if 'timestamp' in stock_data.columns:
                columns_to_drop.append('timestamp')
            if 'symbol' in stock_data.columns:
                columns_to_drop.append('symbol')
            
            if columns_to_drop:
                stock_data = stock_data.drop(columns=columns_to_drop)

            # Ensure target column is last
            target_col = "Mid_Price"
            if target_col in stock_data.columns:
                cols = stock_data.columns.tolist()
                cols.remove(target_col)
                cols.append(target_col)
                stock_data = stock_data[cols]

            # Load model and scaler
            model_path = get_model_path(selected_stock)
            scaler_path = get_scaler_path(selected_stock)

            if os.path.exists(model_path) and os.path.exists(scaler_path):
                try:
                    # Load model without compilation (prediction only, no training)
                    model = load_model(model_path, compile=False)
                    
                    # Load scaler
                    with open(scaler_path, 'rb') as f:
                        scaler = pickle.load(f)

                    # Scale all features
                    df_scaled = scaler.transform(stock_data.values)

                    # Take last WINDOW_SIZE rows for prediction
                    x_latest = df_scaled[-WINDOW_SIZE:]
                    x_latest = x_latest.reshape(1, WINDOW_SIZE, df_scaled.shape[1])

                    # Predict
                    y_pred_scaled = model.predict(x_latest, verbose=0)

                    # Inverse transform
                    dummy = np.zeros((1, df_scaled.shape[1]))
                    dummy[0, -1] = y_pred_scaled.flatten()[0]
                    predicted_price = scaler.inverse_transform(dummy)[0, -1]

                    prediction = round(predicted_price, 2)
                except Exception as e:
                    prediction = f"⚠️ Error during prediction: {str(e)}"
            else:
                prediction = f"⚠️ Model or scaler not found for {selected_stock}"

            # Create figure using latest intraday data only
            if len(plot_timestamps) > 0:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=plot_timestamps,
                    y=plot_closes,
                    mode='lines+markers',
                    name=selected_stock,
                    line=dict(color='#2E86DE', width=2),
                    marker=dict(size=4)
                ))
                
                # Update layout
                fig.update_layout(
                    title=f"{selected_stock} Intraday Close Price ({plot_date})",
                    xaxis_title="Time",
                    yaxis_title="Price (USD)",
                    hovermode='x unified',
                    template='plotly_white',
                    height=500
                )

                # Convert to JSON
                graphJSON = fig.to_json()
            else:
                graphJSON = None
                if prediction != "⚠️ Not enough data for prediction":
                    prediction = "⚠️ No intraday CSV file found"

    return render_template(
        "index.html",
        stocks=STOCKS,
        selected_stock=selected_stock,
        graphJSON=graphJSON,
        prediction=prediction,
        last_close=last_close
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)