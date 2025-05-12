"""
Configuration settings for the project.
"""
import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'northwind'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', '12345')  # Empty default password
}

# Model configuration
MODEL_CONFIG = {
    'input_dim': 20,  # Number of input features
    'hidden_layers': [64, 32, 16],  # Hidden layer sizes
    'dropout_rate': 0.3,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100,
    'early_stopping_patience': 10,
    'validation_split': 0.2
}

# Feature engineering configuration
FEATURE_CONFIG = {
    'time_windows': [7, 30, 90, 180],  # Days for rolling features
    'lag_periods': [1, 3, 7, 14],  # Days for lag features
    'percentiles': [0.25, 0.5, 0.75],  # Percentiles for feature calculation
    'min_purchase_count': 3,  # Minimum purchases for customer analysis
    'target_categories': [1, 2, 3, 4, 5, 6, 7, 8]  # Categories to predict
}

# Data processing configuration
DATA_CONFIG = {
    'train_test_split': 0.2,
    'random_state': 42,
    'missing_value_strategy': 'mean',  # Options: 'mean', 'median', 'mode', 'drop'
    'feature_scaling': 'standard',  # Options: 'standard', 'minmax', 'robust'
    'categorical_encoding': 'onehot'  # Options: 'onehot', 'label', 'target'
}

# Evaluation configuration
EVAL_CONFIG = {
    'metrics': ['accuracy', 'precision', 'recall', 'f1', 'auc'],
    'threshold': 0.5,
    'cv_folds': 5,
    'confidence_threshold': 0.8
}

# Path configuration
PATH_CONFIG = {
    'data_dir': 'data',
    'raw_data_dir': 'data/raw',
    'processed_data_dir': 'data/processed',
    'model_dir': 'models',
    'report_dir': 'reports',
    'log_dir': 'logs'
}

# Logging configuration
LOG_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S'
}

def get_config() -> Dict[str, Any]:
    """
    Get all configuration settings.
    
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    return {
        'db': DB_CONFIG,
        'model': MODEL_CONFIG,
        'feature': FEATURE_CONFIG,
        'data': DATA_CONFIG,
        'eval': EVAL_CONFIG,
        'path': PATH_CONFIG,
        'log': LOG_CONFIG
    } 