"""
Metrics Viewer - View and analyze model training and prediction metrics
"""
import json
import os
from datetime import datetime
from pathlib import Path

MODELS_DIR = "models"


def load_all_metrics():
    """Load metrics for all stocks."""
    metrics_files = list(Path(MODELS_DIR).glob("*_metrics.json"))
    all_metrics = {}
    
    for metrics_file in metrics_files:
        symbol = metrics_file.stem.replace("_metrics", "")
        with open(metrics_file, 'r') as f:
            all_metrics[symbol] = json.load(f)
    
    return all_metrics


def display_latest_metrics(symbol, metrics_data):
    """Display the latest metrics for a stock."""
    print(f"\n{'=' * 60}")
    print(f"Stock: {symbol}")
    print(f"{'=' * 60}")
    
    history = metrics_data.get("history", [])
    
    if not history:
        print("No metrics available")
        return
    
    # Find latest training and prediction
    latest_training = None
    latest_prediction = None
    
    for entry in reversed(history):
        if entry["operation"] in ["train", "retrain"] and latest_training is None:
            latest_training = entry
        if entry["operation"] == "prediction" and latest_prediction is None:
            latest_prediction = entry
        if latest_training and latest_prediction:
            break
    
    # Display training metrics
    if latest_training:
        print(f"\nLatest Training ({latest_training['timestamp']}):")
        print(f"  Operation: {latest_training['operation']}")
        print(f"  Samples: {latest_training['samples']['train']} train, {latest_training['samples']['test']} test")
        print(f"  Epochs Run: {latest_training['epochs_run']}")
        print(f"\n  Performance Metrics:")
        training_metrics = latest_training['training']
        print(f"    Train Loss (MSE): {training_metrics['train_loss_mse']:.6f}")
        print(f"    Train MAE: {training_metrics['train_mae']:.6f}")
        print(f"    Train R²: {training_metrics['train_r2']:.6f}")
        print(f"    Test Loss (MSE): {training_metrics['test_loss_mse']:.6f}")
        print(f"    Test MAE: {training_metrics['test_mae']:.6f}")
        print(f"    Test RMSE: {training_metrics['test_rmse']:.6f}")
        print(f"    Test R²: {training_metrics['test_r2']:.6f}")
    
    # Display prediction metrics
    if latest_prediction:
        print(f"\nLatest Prediction ({latest_prediction['timestamp']}):")
        print(f"  Last Actual Close: ${latest_prediction['last_actual_close']:.2f}")
        print(f"  Predicted Next Price: ${latest_prediction['predicted_next_price']:.2f}")
        print(f"  Price Change: ${latest_prediction['price_change']:.2f} ({latest_prediction['price_change_percent']:.2f}%)")


def display_metrics_history(symbol, metrics_data, operation_type=None):
    """Display historical metrics for a stock."""
    print(f"\n{'=' * 60}")
    print(f"Stock: {symbol} - Historical Metrics")
    print(f"{'=' * 60}")
    
    history = metrics_data.get("history", [])
    
    if not history:
        print("No metrics available")
        return
    
    # Filter by operation type if specified
    if operation_type:
        history = [h for h in history if h["operation"] == operation_type]
    
    if not history:
        print(f"No {operation_type} metrics available")
        return
    
    print(f"\nTotal entries: {len(history)}")
    print(f"\n{'Date':<20} {'Time':<10} {'Operation':<12} {'Key Metric':<40}")
    print("-" * 82)
    
    for entry in history:
        timestamp = entry.get("timestamp", "N/A")
        date_part = timestamp.split(" ")[0] if " " in timestamp else timestamp
        time_part = timestamp.split(" ")[1] if " " in timestamp else ""
        operation = entry["operation"]
        
        # Determine key metric based on operation
        if operation in ["train", "retrain"]:
            key_metric = f"Test R²: {entry['training']['test_r2']:.4f}, RMSE: {entry['training']['test_rmse']:.6f}"
        elif operation == "prediction":
            key_metric = f"Price: ${entry['predicted_next_price']:.2f}, Change: {entry['price_change_percent']:.2f}%"
        else:
            key_metric = "N/A"
        
        print(f"{date_part:<20} {time_part:<10} {operation:<12} {key_metric:<40}")


def display_all_summary():
    """Display summary of all stocks."""
    all_metrics = load_all_metrics()
    
    print("\n" + "=" * 80)
    print("MODEL METRICS SUMMARY - ALL STOCKS")
    print("=" * 80)
    
    if not all_metrics:
        print("No metrics found!")
        return
    
    print(f"\nTotal Stocks: {len(all_metrics)}")
    print(f"\n{'Symbol':<10} {'Total Entries':<15} {'Last Training':<20} {'Last Prediction':<20}")
    print("-" * 80)
    
    for symbol, metrics_data in sorted(all_metrics.items()):
        history = metrics_data.get("history", [])
        total_entries = len(history)
        
        # Find last training and prediction
        last_training = None
        last_prediction = None
        
        for entry in reversed(history):
            if entry["operation"] in ["train", "retrain"] and last_training is None:
                last_training = entry.get("date", "N/A")
            if entry["operation"] == "prediction" and last_prediction is None:
                last_prediction = entry.get("date", "N/A")
        
        last_training = last_training or "N/A"
        last_prediction = last_prediction or "N/A"
        
        print(f"{symbol:<10} {total_entries:<15} {last_training:<20} {last_prediction:<20}")


def main():
    """Main function to display metrics."""
    import sys
    
    all_metrics = load_all_metrics()
    
    if not all_metrics:
        print("No metrics found in the models directory!")
        return
    
    # Display summary
    display_all_summary()
    
    # Display detailed metrics for each stock
    for symbol, metrics_data in sorted(all_metrics.items()):
        display_latest_metrics(symbol, metrics_data)
    
    # Optionally display full history
    print("\n" + "=" * 80)
    show_history = input("\nShow full history for all stocks? (y/n): ").strip().lower()
    
    if show_history == 'y':
        for symbol, metrics_data in sorted(all_metrics.items()):
            display_metrics_history(symbol, metrics_data)


if __name__ == "__main__":
    main()
