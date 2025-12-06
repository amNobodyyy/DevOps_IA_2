import yfinance as yf
import pandas as pd
import json
import os
from datetime import datetime, timedelta

DATA_DIR = "data"
SYMBOLS_FILE = "symbols.json"

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


def fetch_intraday_yfinance(symbol, interval="2m", period=None, start=None, end=None):
    """
    Fetch intraday data for a symbol using yfinance.
    
    Args:
        symbol: Stock ticker symbol
        interval: '1m', '5m', '15m', '30m', '60m', '1h'
        period: '1d', '5d', '1mo', '2mo' (use either period OR start/end)
        start/end: datetime objects or strings for date range
    
    Returns:
        DataFrame with OHLCV data or None if failed
    """
    print(f"  Fetching from yfinance: {symbol} (interval={interval})")
    
    try:
        ticker = yf.Ticker(symbol)
        
        if period:
            df = ticker.history(period=period, interval=interval)
        else:
            df = ticker.history(start=start, end=end, interval=interval)
        
        if df.empty:
            print(f"  WARNING {symbol}: No data returned from yfinance")
            return None
        
        # Rename columns to match existing format (lowercase)
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        # Keep only the columns we need
        cols_to_keep = ['open', 'high', 'low', 'close', 'volume']
        df = df[[col for col in cols_to_keep if col in df.columns]]
        
        # Add symbol column
        df['symbol'] = symbol
        
        print(f"  ✓ Successfully fetched {len(df)} records")
        return df
        
    except Exception as e:
        print(f"  ERROR fetching {symbol}: {str(e)}")
        return None


def fetch_new_stock_data(symbol):
    """
    Fetch data for a NEW stock: 5-minute interval for last 60 days.
    yfinance allows max 60 days for 5m interval.
    """
    print(f"Fetching 60 days of 2-minute data for NEW stock: {symbol}...")
    df = fetch_intraday_yfinance(symbol, interval="2m", period="60d")
    return df


def fetch_existing_stock_data(symbol):
    """
    Fetch data for an EXISTING stock: 1-minute interval for the previous trading day.
    This is used for daily retraining.
    """
    print(f"Fetching previous day's 1-minute data for EXISTING stock: {symbol}...")
    
    # Calculate previous trading day
    today = datetime.now()
    
    # Go back to find the previous trading day (skip weekends)
    days_back = 1
    prev_day = today - timedelta(days=days_back)
    
    # Skip weekends (Saturday=5, Sunday=6)
    while prev_day.weekday() >= 5:
        days_back += 1
        prev_day = today - timedelta(days=days_back)
    
    # For 1m data, we need to fetch a small period
    # yfinance 1m data is available for last 7 days only
    # We'll fetch last 5 days and filter to the previous trading day
    df = fetch_intraday_yfinance(symbol, interval="1m", period="5d")
    
    if df is None or df.empty:
        return None
    
    # Filter to only the previous trading day's data
    prev_day_str = prev_day.strftime("%Y-%m-%d")
    df_filtered = df[df.index.strftime("%Y-%m-%d") == prev_day_str].copy()
    
    if df_filtered.empty:
        # Try to get the most recent day's data available
        available_dates = df.index.strftime("%Y-%m-%d").unique()
        if len(available_dates) > 0:
            latest_date = sorted(available_dates)[-1]
            print(f"  No data for {prev_day_str}, using latest available: {latest_date}")
            df_filtered = df[df.index.strftime("%Y-%m-%d") == latest_date].copy()
    
    if df_filtered.empty:
        print(f"  WARNING: No data available for previous trading day")
        return None
    
    print(f"  Filtered to {len(df_filtered)} records for date: {df_filtered.index[0].strftime('%Y-%m-%d')}")
    return df_filtered


def save_stock_data(symbol, df, data_type="history"):
    """
    Save stock data to CSV.
    data_type: 'history' for new stock historical data or 'daily' for daily updates
    """
    if df is None or len(df) == 0:
        return False

    # Reset index to make timestamp a column
    df_to_save = df.copy()
    
    # Check if index contains datetime (timestamp)
    if isinstance(df_to_save.index, pd.DatetimeIndex):
        df_to_save = df_to_save.reset_index()
        # Rename the index column to 'timestamp'
        if 'Datetime' in df_to_save.columns:
            df_to_save.rename(columns={'Datetime': 'timestamp'}, inplace=True)
        elif 'Date' in df_to_save.columns:
            df_to_save.rename(columns={'Date': 'timestamp'}, inplace=True)
        elif 'index' in df_to_save.columns:
            df_to_save.rename(columns={'index': 'timestamp'}, inplace=True)
        elif df_to_save.columns[0] not in ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol']:
            df_to_save.rename(columns={df_to_save.columns[0]: 'timestamp'}, inplace=True)
    
    # Ensure timestamp column exists and is datetime
    if 'timestamp' in df_to_save.columns:
        df_to_save["timestamp"] = pd.to_datetime(df_to_save["timestamp"])
        # Remove timezone info to match Alpha Vantage format
        if df_to_save["timestamp"].dt.tz is not None:
            df_to_save["timestamp"] = df_to_save["timestamp"].dt.tz_localize(None)

    # Convert numeric columns
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in df_to_save.columns:
            df_to_save[col] = pd.to_numeric(df_to_save[col], errors="coerce")

    # Sort by timestamp ascending (oldest first)
    if 'timestamp' in df_to_save.columns:
        df_to_save = df_to_save.sort_values("timestamp").reset_index(drop=True)

    if data_type == "history":
        # For new stocks: save as {SYMBOL}_history.csv with 5m interval data
        filepath = os.path.join(DATA_DIR, f"{symbol}_history.csv")
        df_to_save.to_csv(filepath, index=False)
        print(f"Saved historical data: {filepath} ({len(df_to_save)} records)")
    elif data_type == "daily":
        # For existing stocks: save as {SYMBOL}_{DATE}.csv with 1m interval data
        if 'timestamp' in df_to_save.columns and len(df_to_save) > 0:
            date_str = df_to_save['timestamp'].iloc[0].strftime("%Y-%m-%d")
        else:
            date_str = datetime.now().strftime("%Y-%m-%d")
        filepath = os.path.join(DATA_DIR, f"{symbol}_{date_str}.csv")
        df_to_save.to_csv(filepath, index=False)
        print(f"Saved daily data: {filepath} ({len(df_to_save)} records)")

    return True


def handle_new_stock(symbol):
    """Handle a NEW stock: fetch 60 days of 5-minute data."""
    print(f"\n{'='*40}")
    print(f"NEW STOCK: {symbol}")
    print(f"{'='*40}")
    
    # Fetch 60 days of 5-minute data
    hist_df = fetch_new_stock_data(symbol)
    if hist_df is not None:
        save_stock_data(symbol, hist_df, data_type="history")
        return True
    else:
        print(f"ERROR: Failed to fetch data for {symbol}")
        return False


def handle_existing_stock(symbol):
    """Handle an EXISTING stock: fetch previous day's 1-minute data for retraining."""
    print(f"\n{'='*40}")
    print(f"EXISTING STOCK: {symbol}")
    print(f"{'='*40}")
    
    daily_df = fetch_existing_stock_data(symbol)
    if daily_df is not None:
        save_stock_data(symbol, daily_df, data_type="daily")
        return True
    else:
        print(f"WARNING: No daily data available for {symbol}")
        return False


def main():
    """Main data fetching orchestration."""
    print("=" * 60)
    print("Stock Intraday Data Fetcher - MLOps Pipeline (yfinance)")
    print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print("\nData fetching strategy:")
    print("  - NEW stocks: 5-minute interval, last 60 days")
    print("  - EXISTING stocks: 1-minute interval, previous trading day")

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

    # Process new stocks (5m interval, 2 months)
    if new_stocks:
        print(f"\n{'#'*60}")
        print(f"Processing {len(new_stocks)} NEW stocks (2m interval, 60 days)...")
        print(f"{'#'*60}")
        for symbol in new_stocks:
            if handle_new_stock(symbol):
                success_count += 1
            else:
                failed_stocks.append(symbol)

    # Process existing stocks (1m interval, previous day)
    if existing_stocks:
        print(f"\n{'#'*60}")
        print(f"Processing {len(existing_stocks)} EXISTING stocks (1m interval, previous day)...")
        print(f"{'#'*60}")
        for symbol in sorted(existing_stocks):
            if handle_existing_stock(symbol):
                success_count += 1
            else:
                failed_stocks.append(symbol)

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
        raise SystemExit(1)
    
    # If some stocks succeeded, consider it a success
    if failed_stocks:
        print(f"\n⚠️ WARNING: {len(failed_stocks)} stock(s) failed to update, but continuing...")
        print("This run is considered a SUCCESS (partial update).")
    else:
        print("\n✅ SUCCESS: All stocks updated successfully!")


if __name__ == "__main__":
    main()
