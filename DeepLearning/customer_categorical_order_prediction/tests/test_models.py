"""
Tests for model functions.
"""
import pytest
import numpy as np
import tensorflow as tf
from src.models.neural_network import CustomerCategoryPredictor
from src.models.model_evaluation import (
    plot_training_history,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_confusion_matrix,
    generate_evaluation_report,
    analyze_feature_importance
)

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    return X, y

@pytest.fixture
def sample_model(sample_data):
    """Create a sample model for testing."""
    X, _ = sample_data
    model = CustomerCategoryPredictor(input_dim=X.shape[1])
    return model

def test_model_initialization(sample_data):
    """Test model initialization."""
    X, _ = sample_data
    model = CustomerCategoryPredictor(input_dim=X.shape[1])
    
    assert isinstance(model.model, tf.keras.Sequential)
    assert model.history is None

def test_model_training(sample_model, sample_data):
    """Test model training."""
    X, y = sample_data
    X_train, X_val = X[:80], X[80:]
    y_train, y_val = y[:80], y[80:]
    
    history = sample_model.train(
        X_train, y_train,
        X_val, y_val,
        batch_size=32,
        epochs=2
    )
    
    assert isinstance(history, dict)
    assert 'loss' in history
    assert 'accuracy' in history

def test_model_evaluation(sample_model, sample_data):
    """Test model evaluation."""
    X, y = sample_data
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]
    
    # Train model
    sample_model.train(X_train, y_train, X_test, y_test, epochs=2)
    
    # Evaluate model
    metrics = sample_model.evaluate(X_test, y_test)
    
    assert isinstance(metrics, dict)
    assert 'loss' in metrics
    assert 'accuracy' in metrics

def test_model_prediction(sample_model, sample_data):
    """Test model prediction."""
    X, _ = sample_data
    predictions = sample_model.predict(X)
    
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape[0] == X.shape[0]
    assert predictions.shape[1] == 1

def test_model_save_load(sample_model, sample_data, tmp_path):
    """Test model saving and loading."""
    X, y = sample_data
    X_train, X_val = X[:80], X[80:]
    y_train, y_val = y[:80], y[80:]
    
    # Train model
    sample_model.train(X_train, y_train, X_val, y_val, epochs=2)
    
    # Save model
    save_path = tmp_path / "test_model.h5"
    sample_model.save_model(str(save_path))
    
    # Load model
    loaded_model = CustomerCategoryPredictor.load_model(str(save_path))
    
    # Compare predictions
    original_preds = sample_model.predict(X)
    loaded_preds = loaded_model.predict(X)
    
    np.testing.assert_array_almost_equal(original_preds, loaded_preds)

def test_plot_training_history(sample_model, sample_data, tmp_path):
    """Test training history plotting."""
    X, y = sample_data
    X_train, X_val = X[:80], X[80:]
    y_train, y_val = y[:80], y[80:]
    
    # Train model
    history = sample_model.train(X_train, y_train, X_val, y_val, epochs=2)
    
    # Plot history
    save_path = tmp_path / "history.png"
    plot_training_history(history, str(save_path))
    
    assert save_path.exists()

def test_plot_roc_curve(sample_model, sample_data, tmp_path):
    """Test ROC curve plotting."""
    X, y = sample_data
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]
    
    # Train model
    sample_model.train(X_train, y_train, X_test, y_test, epochs=2)
    
    # Get predictions
    y_pred = sample_model.predict(X_test)
    
    # Plot ROC curve
    save_path = tmp_path / "roc.png"
    plot_roc_curve(y_test, y_pred, str(save_path))
    
    assert save_path.exists()

def test_plot_confusion_matrix(sample_model, sample_data, tmp_path):
    """Test confusion matrix plotting."""
    X, y = sample_data
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]
    
    # Train model
    sample_model.train(X_train, y_train, X_test, y_test, epochs=2)
    
    # Get predictions
    y_pred = sample_model.predict(X_test)
    
    # Plot confusion matrix
    save_path = tmp_path / "confusion.png"
    plot_confusion_matrix(y_test, y_pred, threshold=0.5, save_path=str(save_path))
    
    assert save_path.exists()

def test_generate_evaluation_report(sample_model, sample_data, tmp_path):
    """Test evaluation report generation."""
    X, y = sample_data
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]
    
    # Train model
    sample_model.train(X_train, y_train, X_test, y_test, epochs=2)
    
    # Get predictions
    y_pred = sample_model.predict(X_test)
    
    # Generate report
    save_dir = tmp_path / "reports"
    report = generate_evaluation_report(
        y_test, y_pred,
        threshold=0.5,
        save_dir=str(save_dir)
    )
    
    assert isinstance(report, dict)
    assert save_dir.exists()
    assert (save_dir / "metrics.json").exists()

def test_analyze_feature_importance(sample_model, sample_data, tmp_path):
    """Test feature importance analysis."""
    X, y = sample_data
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    # Train model
    sample_model.train(X[:80], y[:80], X[80:], y[80:], epochs=2)
    
    # Analyze feature importance
    save_path = tmp_path / "importance.png"
    importance_df = analyze_feature_importance(
        sample_model.model,
        feature_names,
        save_path=str(save_path)
    )
    
    assert isinstance(importance_df, pd.DataFrame)
    assert 'feature' in importance_df.columns
    assert 'importance' in importance_df.columns
    assert save_path.exists() 