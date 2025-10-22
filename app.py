from flask import Flask, render_template, request
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json

app = Flask(__name__)

# Load today's CSV
CSV_FILE = "tech_stocks_intraday_today.csv"
df = pd.read_csv(CSV_FILE, parse_dates=["timestamp"])

# Get list of unique symbols
STOCKS = df["symbol"].unique().tolist()

@app.route("/", methods=["GET", "POST"])
def home():
    selected_stock = None
    graphJSON = None

    if request.method == "POST":
        selected_stock = request.form.get("stock")
        stock_data = df[df["symbol"] == selected_stock].sort_values(by="timestamp").reset_index(drop=True)
        
        # Create figure using graph_objects for better control
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=stock_data["timestamp"].tolist(),
            y=stock_data["close"].tolist(),
            mode='lines+markers',
            name=selected_stock
        ))
        
        # Update layout
        fig.update_layout(
            title=f"{selected_stock} Intraday Close Price",
            xaxis_title="Time",
            yaxis_title="Price (USD)",
            hovermode='x unified'
        )

        # Convert to JSON using Plotly's to_json method
        graphJSON = fig.to_json()

    return render_template(
        "index.html",
        stocks=STOCKS,
        selected_stock=selected_stock,
        graphJSON=graphJSON
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)