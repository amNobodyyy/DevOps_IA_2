import requests
import pandas as pd
from datetime import datetime
import time

API_KEY = "AWBE6N6H7MIU812S"

# Top 10 Tech Stocks 
symbols = ["AAPL", "MSFT", "NVDA"]

def fetch_intraday(symbol, interval="60min"):
    """Fetch all intraday (hourly) data for a symbol."""
    url = (
        f"https://www.alphavantage.co/query?"
        f"function=TIME_SERIES_INTRADAY&symbol={symbol}"
        f"&interval={interval}&outputsize=full&apikey={API_KEY}"
    )
    response = requests.get(url)
    data = response.json()

    key = f"Time Series ({interval})"
    if key not in data:
        print(f"⚠️ {symbol}: {data.get('Note', data.get('Error Message', 'No intraday data'))}")
        return None

    df = pd.DataFrame(data[key]).T
    df.index = pd.to_datetime(df.index)
    df = df.rename(columns=lambda x: x.split(". ")[1])
    df = df.sort_index()
    df["symbol"] = symbol
    return df

def fetch_all_symbols(symbols):
    all_data = []
    for symbol in symbols:
        print(f"Fetching hourly data for {symbol}...")
        df = fetch_intraday(symbol)
        if df is not None:
            all_data.append(df)
        time.sleep(5)  # 5 requests per minute (free tier)
    return all_data

def filter_latest_day_data(df):
    # Filter DataFrame to only include data from the latest day.
    latest_data = []
    for sym, group in df.groupby("symbol"):
        latest_date = group['timestamp'].dt.date.max()
        latest_data.append(group[group['timestamp'].dt.date == latest_date])
    return pd.concat(latest_data)

# ---- MAIN ----
all_data = fetch_all_symbols(symbols)

if all_data:
    full_df = pd.concat(all_data)
    full_df = full_df.rename_axis("timestamp").reset_index()
    
    # Ensure timestamp column is datetime
    full_df['timestamp'] = pd.to_datetime(full_df['timestamp'])

    # Convert numeric columns
    numeric_cols = ["open", "high", "low", "close", "volume"]
    full_df[numeric_cols] = full_df[numeric_cols].apply(pd.to_numeric)
    
    # Sort by symbol and timestamp descending
    full_df = full_df.sort_values(by=["symbol", "timestamp"], ascending=[True, False]).reset_index(drop=True)

    # Debug: print column types
    print("\n--- Column Types ---")
    for col in full_df.columns:
        print(f"{col}: {full_df[col].dtype}")
        
    # Optional: Filter to only the latest day's data
    full_df = filter_latest_day_data(full_df)

    full_df.to_csv("tech_stocks_intraday_today.csv", index=False)
    print("\n✅ Saved hourly intraday data for today → tech_stocks_intraday_today.csv")
    print(full_df.head())
else:
    print("❌ No data fetched.")
