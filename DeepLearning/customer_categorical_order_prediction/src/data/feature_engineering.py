"""
Feature engineering module for customer category prediction.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from .database import get_customer_category_data

def create_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create customer-level features.
    
    Args:
        df (pd.DataFrame): Input DataFrame with customer data
        
    Returns:
        pd.DataFrame: DataFrame with customer features
    """
    # Convert order_date to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df['order_date']):
        df['order_date'] = pd.to_datetime(df['order_date'])
    
    # Group by customer and calculate metrics
    customer_features = df.groupby('customer_id').agg({
        'order_id': 'count',
        'total_amount': ['sum', 'mean', 'std'],
        'order_date': ['min', 'max'],
        'category_id': 'nunique'  # Number of unique categories purchased
    }).reset_index()
    
    # Flatten column names
    customer_features.columns = ['customer_id', 'total_orders',
                               'total_spent', 'avg_order_value',
                               'std_order_value', 'first_order_date',
                               'last_order_date', 'unique_categories']
    
    # Calculate time-based features
    customer_features['customer_lifetime'] = (
        customer_features['last_order_date'] - customer_features['first_order_date']
    ).dt.days
    
    customer_features['avg_days_between_orders'] = (
        customer_features['customer_lifetime'] / customer_features['total_orders']
    )
    
    # Calculate category diversity
    customer_features['category_diversity'] = (
        customer_features['unique_categories'] / customer_features['total_orders']
    )
    
    return customer_features

def create_category_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create category-level features.
    
    Args:
        df (pd.DataFrame): Input DataFrame with category data
        
    Returns:
        pd.DataFrame: DataFrame with category features
    """
    # Convert order_date to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df['order_date']):
        df['order_date'] = pd.to_datetime(df['order_date'])
    
    # Group by customer and category
    category_features = df.groupby(['customer_id', 'category_id']).agg({
        'order_id': 'count',
        'total_amount': ['sum', 'mean'],
        'order_date': ['min', 'max']
    }).reset_index()
    
    # Flatten column names
    category_features.columns = ['customer_id', 'category_id',
                               'category_orders', 'category_spent',
                               'avg_category_order', 'first_category_order',
                               'last_category_order']
    
    # Calculate category-specific metrics
    category_features['category_lifetime'] = (
        category_features['last_category_order'] - category_features['first_category_order']
    ).dt.days
    
    category_features['category_order_frequency'] = (
        category_features['category_lifetime'] / category_features['category_orders']
    )
    
    return category_features

def create_time_based_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features.
    
    Args:
        df (pd.DataFrame): Input DataFrame with temporal data
        
    Returns:
        pd.DataFrame: DataFrame with time-based features
    """
    # Convert order_date to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df['order_date']):
        df['order_date'] = pd.to_datetime(df['order_date'])
    
    # Extract time components
    df['order_year'] = df['order_date'].dt.year
    df['order_month'] = df['order_date'].dt.month
    df['order_day'] = df['order_date'].dt.day
    df['order_dayofweek'] = df['order_date'].dt.dayofweek
    df['order_quarter'] = df['order_date'].dt.quarter
    
    # Calculate time since last order
    df['days_since_last_order'] = df.groupby('customer_id')['order_date'].diff().dt.days
    
    return df

def calculate_customer_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate customer-level metrics.
    
    Args:
        df (pd.DataFrame): Input DataFrame with customer data
        
    Returns:
        pd.DataFrame: DataFrame with customer metrics
    """
    # Convert order_date to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df['order_date']):
        df['order_date'] = pd.to_datetime(df['order_date'])
    
    # Calculate RFM metrics
    current_date = df['order_date'].max()
    
    rfm = df.groupby('customer_id').agg({
        'order_date': lambda x: (current_date - x.max()).days,  # Recency
        'order_id': 'count',  # Frequency
        'total_amount': 'sum'  # Monetary
    }).reset_index()
    
    rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']
    
    # Calculate additional metrics
    rfm['avg_order_value'] = rfm['monetary'] / rfm['frequency']
    rfm['purchase_rate'] = rfm['frequency'] / (
        (current_date - df.groupby('customer_id')['order_date'].min()).dt.days
    )
    
    return rfm

def prepare_model_data(df: pd.DataFrame,
                      target_category: int,
                      test_size: float = 0.2,
                      random_state: int = 42) -> Tuple[np.ndarray, np.ndarray,
                                                     np.ndarray, np.ndarray]:
    """
    Prepare data for model training.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        target_category (int): Target category ID
        test_size (float): Test set size
        random_state (int): Random state for reproducibility
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: X_train, X_test, y_train, y_test
    """
    # Use category_id_x as the main category_id column
    df['target'] = (df['category_id_x'] == target_category).astype(int)
    
    # Select features
    feature_cols = [col for col in df.columns if col not in
                   ['customer_id', 'category_id_x', 'category_id_y', 'order_id', 'order_date',
                    'target', 'first_order_date', 'last_order_date',
                    'first_category_order', 'last_category_order',
                    'company_name', 'category_name']]
    
    X = df[feature_cols]
    y = df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

def handle_missing_values(df: pd.DataFrame,
                         strategy: str = 'mean') -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        strategy (str): Strategy for handling missing values
        
    Returns:
        pd.DataFrame: DataFrame with handled missing values
    """
    if strategy == 'mean':
        return df.fillna(df.mean())
    elif strategy == 'median':
        return df.fillna(df.median())
    elif strategy == 'mode':
        return df.fillna(df.mode().iloc[0])
    elif strategy == 'drop':
        return df.dropna()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def get_train_test_split(df: pd.DataFrame,
                        target_col: str,
                        test_size: float = 0.2,
                        random_state: int = 42) -> Tuple[pd.DataFrame,
                                                       pd.DataFrame,
                                                       pd.Series,
                                                       pd.Series]:
    """
    Split data into training and test sets.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        target_col (str): Target column name
        test_size (float): Test set size
        random_state (int): Random state for reproducibility
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y
    ) 