"""
Neural network model for customer category prediction.
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import AUC
from typing import Tuple, Dict, Any
import json

from src.data.feature_engineering import (
    create_customer_features,
    prepare_model_data,
    get_train_test_split
)

class CustomerCategoryPredictor:
    """
    Neural network model for predicting customer category purchases.
    """
    def __init__(self,
                 input_dim: int,
                 hidden_layers: list = [64, 32, 16],
                 dropout_rate: float = 0.3,
                 learning_rate: float = 0.001):
        """
        Initialize the model.
        
        Args:
            input_dim (int): Number of input features
            hidden_layers (list): List of hidden layer sizes
            dropout_rate (float): Dropout rate for regularization
            learning_rate (float): Learning rate for optimizer
        """
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = self._build_model()
        
    def _build_model(self) -> Sequential:
        """
        Build the neural network model.
        
        Returns:
            Sequential: Compiled Keras model
        """
        model = Sequential()
        
        # Input layer
        model.add(Dense(self.hidden_layers[0], activation='relu', input_dim=self.input_dim))
        model.add(Dropout(self.dropout_rate))
        
        # Hidden layers
        for units in self.hidden_layers[1:]:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(self.dropout_rate))
        
        # Output layer
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', AUC(name='auc')]
        )
        
        return model
    
    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              batch_size: int = 32,
              epochs: int = 100,
              validation_split: float = 0.2) -> dict:
        """
        Train the model.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            batch_size (int): Batch size for training
            epochs (int): Number of training epochs
            validation_split (float): Validation split ratio
            
        Returns:
            dict: Training history
        """
        # Create callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                mode='min'
            ),
            ModelCheckpoint(
                'best_model.keras',
                monitor='val_loss',
                save_best_only=True,
                mode='min'
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        return history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Predicted probabilities
        """
        return self.model.predict(X, verbose=0)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Evaluate the model.
        
        Args:
            X (np.ndarray): Test features
            y (np.ndarray): Test labels
            
        Returns:
            dict: Evaluation metrics
        """
        return dict(zip(self.model.metrics_names,
                       self.model.evaluate(X, y, verbose=0)))
    
    def save(self, filepath: str):
        """
        Save the model.
        
        Args:
            filepath (str): Path to save the model
        """
        # Ensure filepath ends with .keras
        if not filepath.endswith('.keras'):
            filepath = f"{filepath}.keras"
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model without specifying save_format
        self.model.save(filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'CustomerCategoryPredictor':
        """
        Load a saved model.
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            CustomerCategoryPredictor: Loaded model
        """
        # Ensure filepath ends with .keras
        if not filepath.endswith('.keras'):
            filepath = f"{filepath}.keras"
            
        model = load_model(filepath, compile=True)
        instance = cls(input_dim=model.input_shape[1])
        instance.model = model
        return instance

def main():
    """Main training script."""
    # Prepare data
    df = create_customer_features()
    X, y = prepare_model_data(df)
    X_train, X_test, y_train, y_test = get_train_test_split(X, y)
    
    # Split training data into train and validation
    X_train, X_val, y_train, y_val = get_train_test_split(X_train, y_train, test_size=0.2)
    
    # Initialize and train model
    model = CustomerCategoryPredictor(input_dim=X.shape[1])
    history = model.train(X_train, y_train, X_val, y_val)
    
    # Evaluate model
    metrics = model.evaluate(X_test, y_test)
    print("\nTest Metrics:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    
    # Save model
    model.save('models/saved/customer_category_predictor.h5')

if __name__ == '__main__':
    main() 