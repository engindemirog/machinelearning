"""
Helper functions for the project.
"""
import os
import json
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_directories() -> None:
    """
    Create necessary directories for the project.
    """
    directories = [
        'data/raw',
        'data/processed',
        'models',
        'reports',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def save_json(data: Dict[str, Any], filepath: str) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data (Dict[str, Any]): Data to save
        filepath (str): Path to save the file
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        logger.info(f"Saved data to {filepath}")
    except Exception as e:
        logger.error(f"Error saving data to {filepath}: {str(e)}")
        raise

def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load data from a JSON file.
    
    Args:
        filepath (str): Path to the file
        
    Returns:
        Dict[str, Any]: Loaded data
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded data from {filepath}")
        return data
    except Exception as e:
        logger.error(f"Error loading data from {filepath}: {str(e)}")
        raise

def save_dataframe(df: pd.DataFrame, filepath: str) -> None:
    """
    Save DataFrame to a file.
    
    Args:
        df (pd.DataFrame): DataFrame to save
        filepath (str): Path to save the file
    """
    try:
        df.to_csv(filepath, index=False)
        logger.info(f"Saved DataFrame to {filepath}")
    except Exception as e:
        logger.error(f"Error saving DataFrame to {filepath}: {str(e)}")
        raise

def load_dataframe(filepath: str) -> pd.DataFrame:
    """
    Load DataFrame from a file.
    
    Args:
        filepath (str): Path to the file
        
    Returns:
        pd.DataFrame: Loaded DataFrame
    """
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded DataFrame from {filepath}")
        return df
    except Exception as e:
        logger.error(f"Error loading DataFrame from {filepath}: {str(e)}")
        raise

def calculate_time_features(df: pd.DataFrame,
                          date_column: str) -> pd.DataFrame:
    """
    Calculate time-based features from a date column.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        date_column (str): Name of the date column
        
    Returns:
        pd.DataFrame: DataFrame with additional time features
    """
    df = df.copy()
    
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        df[date_column] = pd.to_datetime(df[date_column])
    
    # Extract time components
    df[f'{date_column}_year'] = df[date_column].dt.year
    df[f'{date_column}_month'] = df[date_column].dt.month
    df[f'{date_column}_day'] = df[date_column].dt.day
    df[f'{date_column}_dayofweek'] = df[date_column].dt.dayofweek
    df[f'{date_column}_quarter'] = df[date_column].dt.quarter
    
    return df

def calculate_rolling_features(df: pd.DataFrame,
                             group_col: str,
                             value_col: str,
                             windows: List[int]) -> pd.DataFrame:
    """
    Calculate rolling window features.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        group_col (str): Column to group by
        value_col (str): Column to calculate rolling features for
        windows (List[int]): List of window sizes
        
    Returns:
        pd.DataFrame: DataFrame with rolling features
    """
    df = df.copy()
    
    for window in windows:
        # Calculate rolling mean
        df[f'{value_col}_rolling_mean_{window}'] = (
            df.groupby(group_col)[value_col]
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
        
        # Calculate rolling std
        df[f'{value_col}_rolling_std_{window}'] = (
            df.groupby(group_col)[value_col]
            .transform(lambda x: x.rolling(window, min_periods=1).std())
        )
    
    return df

def calculate_lag_features(df: pd.DataFrame,
                          group_col: str,
                          value_col: str,
                          lags: List[int]) -> pd.DataFrame:
    """
    Calculate lag features.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        group_col (str): Column to group by
        value_col (str): Column to calculate lag features for
        lags (List[int]): List of lag periods
        
    Returns:
        pd.DataFrame: DataFrame with lag features
    """
    df = df.copy()
    
    for lag in lags:
        df[f'{value_col}_lag_{lag}'] = (
            df.groupby(group_col)[value_col]
            .transform(lambda x: x.shift(lag))
        )
    
    return df

def calculate_ratio_features(df: pd.DataFrame,
                           numerator_col: str,
                           denominator_col: str,
                           prefix: str = '') -> pd.DataFrame:
    """
    Calculate ratio features.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        numerator_col (str): Numerator column
        denominator_col (str): Denominator column
        prefix (str): Prefix for the new column name
        
    Returns:
        pd.DataFrame: DataFrame with ratio features
    """
    df = df.copy()
    
    # Calculate ratio
    ratio_col = f'{prefix}ratio' if prefix else 'ratio'
    df[ratio_col] = df[numerator_col] / df[denominator_col]
    
    # Handle division by zero
    df[ratio_col] = df[ratio_col].replace([np.inf, -np.inf], np.nan)
    
    return df

def calculate_percentile_features(df: pd.DataFrame,
                                group_col: str,
                                value_col: str,
                                percentiles: List[float]) -> pd.DataFrame:
    """
    Calculate percentile features.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        group_col (str): Column to group by
        value_col (str): Column to calculate percentiles for
        percentiles (List[float]): List of percentiles to calculate
        
    Returns:
        pd.DataFrame: DataFrame with percentile features
    """
    df = df.copy()
    
    for percentile in percentiles:
        df[f'{value_col}_percentile_{int(percentile*100)}'] = (
            df.groupby(group_col)[value_col]
            .transform(lambda x: x.quantile(percentile))
        )
    
    return df 