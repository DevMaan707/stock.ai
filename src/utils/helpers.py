import os
import json
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Any

def ensure_directory(path: str) -> None:
    """
    Ensure that a directory exists, creating it if necessary
    
    Args:
        path: Directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def save_to_json(data: Any, filename: str) -> bool:
    """
    Save data to a JSON file
    
    Args:
        data: Data to save
        filename: Target filename
        
    Returns:
        Success status
    """
    try:
        ensure_directory(os.path.dirname(filename))
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving to {filename}: {e}")
        return False

def load_from_json(filename: str, default: Any = None) -> Any:
    """
    Load data from a JSON file
    
    Args:
        filename: Source filename
        default: Default value if file doesn't exist
        
    Returns:
        Loaded data or default
    """
    if not os.path.exists(filename):
        return default
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading from {filename}: {e}")
        return default

def save_to_pickle(data: Any, filename: str) -> bool:
    """
    Save data to a pickle file
    
    Args:
        data: Data to save
        filename: Target filename
        
    Returns:
        Success status
    """
    try:
        ensure_directory(os.path.dirname(filename))
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        return True
    except Exception as e:
        print(f"Error saving to {filename}: {e}")
        return False

def load_from_pickle(filename: str, default: Any = None) -> Any:
    """
    Load data from a pickle file
    
    Args:
        filename: Source filename
        default: Default value if file doesn't exist
        
    Returns:
        Loaded data or default
    """
    if not os.path.exists(filename):
        return default
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading from {filename}: {e}")
        return default

def format_price(price: float) -> str:
    """Format price with appropriate precision"""
    if price is None:
        return "N/A"
    if price < 10:
        return f"${price:.4f}"
    elif price < 100:
        return f"${price:.2f}"
    else:
        return f"${price:.2f}"

def calculate_return(start_price: float, end_price: float) -> float:
    """Calculate percentage return between two prices"""
    if start_price == 0:
        return 0
    return ((end_price - start_price) / start_price) * 100

def moving_average(data: np.ndarray, window: int) -> np.ndarray:
    """Calculate simple moving average"""
    return np.convolve(data, np.ones(window)/window, mode='valid')

def exponential_moving_average(data: np.ndarray, span: int) -> np.ndarray:
    """Calculate exponential moving average"""
    alpha = 2 / (span + 1)
    ema = np.zeros_like(data)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
    return ema

def timestamp_to_string(timestamp: float) -> str:
    """Convert timestamp to formatted date string"""
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

def get_current_timestamp() -> float:
    """Get current timestamp"""
    return datetime.now().timestamp()

def resample_data(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """
    Resample time series data to a different interval
    
    Args:
        df: DataFrame with DateTimeIndex
        interval: Pandas resampling string (e.g., '1D', '4H')
        
    Returns:
        Resampled DataFrame
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")
        
    return df.resample(interval).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })