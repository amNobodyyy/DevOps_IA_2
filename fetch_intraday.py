import requests
import pandas as pd
import json
import os
import time
from datetime import datetime
from pathlib import Path

API_KEY = "I0N8RSZ7U6HYEBDB"
DATA_DIR = "data"
SYMBOLS_FILE = "symbols.json"
INTERVAL = "5min"  # Intraday 5-minute data

# Create data directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)


def load_symbols():
    """Load stock symbols from JSON file."""
    with open(SYMBOLS_FILE, "r") as f:
        config = json.load(f)
    return config.get("stocks", [])


def get_existing_stocks():
    """Get list of stocks that already have CSV files in data directory."""
    existing = set()
    for file in os.listdir(DATA_DIR):
        if file.endswith("_history.csv"):
            symbol = file.replace("_history.csv", "")
            existing.add(symbol)
    return existing


def fetch_intraday(symbol, interval="60min"):
    """Fetch all intraday (hourly) data for a symbol from Alpha Vantage."""
    url = (
        f"https://www.alphavantage.co/query?"
        f"function=TIME_SERIES_INTRADAY&symbol={symbol}"
        f"&interval={interval}&outputsize=compact&apikey={API_KEY}"
    )
    
    print(f"  Fetching from API: {symbol} (interval={interval})")
    response = requests.get(url)
    
    # Check HTTP status
    if response.status_code != 200:
        print(f"  ERROR: HTTP {response.status_code}")
        print(f"  Response: {response.text[:200]}")
        return None
    
    data = response.json()
    
    # Debug: Print full response if there's an issue
    key = f"Time Series ({interval})"
    if key not in data:
        print(f"  WARNING {symbol}: API response does not contain expected data")
        print(f"  Available keys: {list(data.keys())}")
        
        # Check for common API errors
        if "Note" in data:
            print(f"  API Note: {data['Note']}")
        if "Error Message" in data:
            print(f"  API Error: {data['Error Message']}")
        if "Information" in data:
            print(f"  API Info: {data['Information']}")
        
        # Print full response for debugging
        print(f"  Full API Response: {json.dumps(data, indent=2)[:500]}")
        return None

    df = pd.DataFrame(data[key]).T
    df.index = pd.to_datetime(df.index)
    df = df.rename(columns=lambda x: x.split(". ")[1])
    df = df.sort_index()
    df["symbol"] = symbol
    print(f"  ✓ Successfully fetched {len(df)} records")
    return df


def fetch_all_historical_data(symbol):
    """Fetch all available intraday data for a new stock."""
    print(f"Fetching ALL historical intraday ({INTERVAL}) data for {symbol}...")
    df = fetch_intraday(symbol, interval=INTERVAL)
    return df


def fetch_today_data(symbol):
    """Fetch today's intraday data for an existing stock."""
    print(f"Fetching today's intraday ({INTERVAL}) data for {symbol}...")
    df = fetch_intraday(symbol, interval=INTERVAL)
    if df is None:
        return None

    # Filter to only today's data
    latest_date = df.index.date.max()
    today_df = df[df.index.date == latest_date].copy()
    return today_df if len(today_df) > 0 else None


def save_stock_data(symbol, df, data_type="history"):
    """
    Save stock data to CSV.
    data_type: 'history' for all historical data or 'today' for daily updates
    """
    if df is None or len(df) == 0:
        return False

    # Reset index to make timestamp a column
    df_to_save = df.copy()
    
    # Check if index contains datetime (timestamp)
    if isinstance(df_to_save.index, pd.DatetimeIndex):
        df_to_save = df_to_save.reset_index()
        # Rename the index column to 'timestamp' if it doesn't have a name
        if df_to_save.columns[0] == 'index':
            df_to_save.rename(columns={'index': 'timestamp'}, inplace=True)
    elif 'timestamp' not in df_to_save.columns:
        # If no timestamp column and index is not datetime, use index as timestamp
        df_to_save = df_to_save.reset_index()
        if 'timestamp' not in df_to_save.columns:
            df_to_save.rename(columns={df_to_save.columns[0]: 'timestamp'}, inplace=True)
    
    # Ensure timestamp column is datetime
    if 'timestamp' in df_to_save.columns:
        df_to_save["timestamp"] = pd.to_datetime(df_to_save["timestamp"])

    # Convert numeric columns
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in df_to_save.columns:
            df_to_save[col] = pd.to_numeric(df_to_save[col], errors="coerce")

    # Sort by timestamp ascending (oldest first)
    df_to_save = df_to_save.sort_values("timestamp").reset_index(drop=True)

    if data_type == "history":
        filepath = os.path.join(DATA_DIR, f"{symbol}_history.csv")
        df_to_save.to_csv(filepath, index=False)
        print(f"Saved historical data: {filepath} ({len(df_to_save)} records)")
    elif data_type == "today":
        today_str = datetime.now().strftime("%Y-%m-%d")
        filepath = os.path.join(DATA_DIR, f"{symbol}_{today_str}.csv")
        df_to_save.to_csv(filepath, index=False)
        print(f"Saved today's data: {filepath} ({len(df_to_save)} records)")

    return True


def handle_new_stock(symbol):
    """Handle a new stock: fetch and save all historical + today's intraday data."""
    print(f"\nNEW STOCK: {symbol}")
    
    # Fetch all historical intraday data
    hist_df = fetch_all_historical_data(symbol)
    if hist_df is not None:
        save_stock_data(symbol, hist_df, data_type="history")
    else:
        print(f"ERROR: Failed to fetch historical intraday data for {symbol}")
        return False

    # Also save today's data
    time.sleep(20)  # Rate limiting
    today_df = fetch_today_data(symbol)
    if today_df is not None:
        save_stock_data(symbol, today_df, data_type="today")
    else:
        print(f"WARNING: No today's intraday data available for {symbol}")

    return True


def handle_existing_stock(symbol):
    """Handle an existing stock: fetch and save only today's intraday data."""
    print(f"\nEXISTING STOCK: {symbol}")
    today_df = fetch_today_data(symbol)
    if today_df is not None:
        save_stock_data(symbol, today_df, data_type="today")
        return True
    else:
        print(f"WARNING: No today's intraday data available for {symbol}")
        return False


def main():
    """Main data fetching orchestration."""
    print("=" * 60)
    print("Stock Intraday Data Fetcher - MLOps Pipeline")
    print(f"Interval: {INTERVAL}")
    print(f"Current UTC time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Load symbols from JSON
    symbols = load_symbols()
    existing_stocks = get_existing_stocks()
    new_stocks = [s for s in symbols if s not in existing_stocks]

    print(f"\nConfiguration:")
    print(f"  Total stocks in config: {len(symbols)}")
    print(f"  Existing stock data: {len(existing_stocks)}")
    print(f"  New stocks to fetch: {len(new_stocks)}")

    # Track successes and failures
    success_count = 0
    failed_stocks = []

    # Process new stocks
    if new_stocks:
        print(f"\nProcessing {len(new_stocks)} new stocks...")
        for symbol in new_stocks:
            if handle_new_stock(symbol):
                success_count += 1
            else:
                failed_stocks.append(symbol)
            time.sleep(20)  # Rate limiting

    # Process existing stocks - update with latest day's data
    if existing_stocks:
        print(f"\nUpdating {len(existing_stocks)} existing stocks with latest intraday data...")
        for symbol in sorted(existing_stocks):
            if handle_existing_stock(symbol):
                success_count += 1
            else:
                failed_stocks.append(symbol)
            time.sleep(20)  # Rate limiting

    print("\n" + "=" * 60)
    print("Intraday data fetching completed!")
    print(f"  Successfully updated: {success_count}/{len(symbols)} stocks")
    
    if failed_stocks:
        print(f"  Failed stocks: {', '.join(failed_stocks)}")
    
    print("=" * 60)
    
    # Exit with error code ONLY if no data was successfully saved
    if success_count == 0:
        print("\n❌ ERROR: No data was saved for any stock!")
        print("This run is considered a FAILURE.")
        raise SystemExit(1)  # Exit with error code 1
    
    # If some stocks succeeded, consider it a success (even if some failed)
    if failed_stocks:
        print(f"\n  WARNING: {len(failed_stocks)} stock(s) failed to update, but continuing...")
        print("This run is considered a SUCCESS (partial update).")
    else:
        print("\n✅ SUCCESS: All stocks updated successfully!")
    


if __name__ == "__main__":
    main()
