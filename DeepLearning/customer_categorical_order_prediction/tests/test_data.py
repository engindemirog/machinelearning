"""
Tests for data processing functions.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.data.feature_engineering import (
    create_customer_features,
    prepare_model_data,
    get_train_test_split
)
from src.utils.helpers import (
    handle_missing_values,
    calculate_customer_metrics,
    create_time_based_features
)

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    data = {
        'customer_id': ['C1', 'C1', 'C2', 'C2', 'C3'],
        'category_name': ['A', 'B', 'A', 'C', 'B'],
        'purchase_count': [2, 1, 3, 1, 2],
        'total_spent': [100, 50, 150, 75, 80],
        'last_purchase_date': [
            datetime.now() - timedelta(days=x)
            for x in [1, 2, 3, 4, 5]
        ]
    }
    return pd.DataFrame(data)

def test_create_customer_features(sample_data):
    """Test customer feature creation."""
    df = create_customer_features()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert 'category_spend_ratio' in df.columns
    assert 'category_purchase_ratio' in df.columns

def test_prepare_model_data(sample_data):
    """Test model data preparation."""
    X, y = prepare_model_data(sample_data)
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert len(X) == len(y)
    assert X.shape[1] > 0

def test_get_train_test_split(sample_data):
    """Test train-test split function."""
    X, y = prepare_model_data(sample_data)
    X_train, X_test, y_train, y_test = get_train_test_split(X, y, test_size=0.2)
    
    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)
    assert X_train.shape[1] == X_test.shape[1]

def test_handle_missing_values(sample_data):
    """Test missing value handling."""
    # Add some missing values
    sample_data.loc[0, 'total_spent'] = np.nan
    sample_data.loc[1, 'category_name'] = None
    
    # Test different strategies
    df_mean = handle_missing_values(sample_data, strategy='mean')
    df_median = handle_missing_values(sample_data, strategy='median')
    df_zero = handle_missing_values(sample_data, strategy='zero')
    
    assert not df_mean.isnull().any().any()
    assert not df_median.isnull().any().any()
    assert not df_zero.isnull().any().any()

def test_calculate_customer_metrics(sample_data):
    """Test customer metrics calculation."""
    metrics = calculate_customer_metrics(
        sample_data,
        customer_id_col='customer_id',
        date_col='last_purchase_date',
        value_col='total_spent'
    )
    
    assert isinstance(metrics, pd.DataFrame)
    assert 'days_since_first_purchase' in metrics.columns
    assert 'days_since_last_purchase' in metrics.columns
    assert 'purchase_frequency' in metrics.columns

def test_create_time_based_features(sample_data):
    """Test time-based feature creation."""
    df = create_time_based_features(sample_data, 'last_purchase_date')
    
    assert 'year' in df.columns
    assert 'month' in df.columns
    assert 'day' in df.columns
    assert 'dayofweek' in df.columns
    assert 'month_sin' in df.columns
    assert 'month_cos' in df.columns 