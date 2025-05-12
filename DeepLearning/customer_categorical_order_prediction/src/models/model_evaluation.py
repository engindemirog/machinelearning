"""
Model evaluation and visualization module.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve
)
from typing import Dict, Any, Tuple
import json

def plot_training_history(history: Dict[str, Any], category_name: str):
    """
    Plot training history metrics.
    
    Args:
        history (Dict[str, Any]): Training history dictionary
        category_name (str): Name of the category being predicted
    """
    # Create reports directory if it doesn't exist
    reports_dir = os.path.join('reports', 'training_history')
    os.makedirs(reports_dir, exist_ok=True)
    
    # Plot metrics
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'Loss - {category_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Accuracy - {category_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(reports_dir, f'training_history_{category_name}.png')
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(y_true: np.ndarray,
                         y_pred: np.ndarray,
                         category_name: str,
                         threshold: float = 0.5):
    """
    Plot confusion matrix.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted probabilities
        category_name (str): Name of the category being predicted
        threshold (float): Classification threshold for converting probabilities to binary predictions
    """
    # Create reports directory if it doesn't exist
    reports_dir = os.path.join('reports', 'confusion_matrices')
    os.makedirs(reports_dir, exist_ok=True)
    
    # Convert probabilities to binary predictions
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred_binary)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {category_name} (threshold={threshold})')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Save plot
    save_path = os.path.join(reports_dir, f'confusion_matrix_{category_name}.png')
    plt.savefig(save_path)
    plt.close()

def plot_roc_curve(y_true: np.ndarray,
                   y_pred_proba: np.ndarray,
                   category_name: str):
    """
    Plot ROC curve.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred_proba (np.ndarray): Predicted probabilities
        category_name (str): Name of the category being predicted
    """
    # Create reports directory if it doesn't exist
    reports_dir = os.path.join('reports', 'roc_curves')
    os.makedirs(reports_dir, exist_ok=True)
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {category_name}')
    plt.legend(loc="lower right")
    
    # Save plot
    save_path = os.path.join(reports_dir, f'roc_curve_{category_name}.png')
    plt.savefig(save_path)
    plt.close()

def plot_precision_recall_curve(y_true: np.ndarray,
                              y_pred_proba: np.ndarray,
                              category_name: str):
    """
    Plot precision-recall curve.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred_proba (np.ndarray): Predicted probabilities
        category_name (str): Name of the category being predicted
    """
    # Create reports directory if it doesn't exist
    reports_dir = os.path.join('reports', 'precision_recall_curves')
    os.makedirs(reports_dir, exist_ok=True)
    
    # Calculate precision-recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    
    # Plot precision-recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {category_name}')
    plt.grid(True)
    
    # Save plot
    save_path = os.path.join(reports_dir, f'precision_recall_curve_{category_name}.png')
    plt.savefig(save_path)
    plt.close()

def generate_evaluation_report(y_true: np.ndarray,
                             y_pred: np.ndarray,
                             y_pred_proba: np.ndarray,
                             category_name: str,
                             threshold: float = 0.5) -> Dict[str, float]:
    """
    Generate comprehensive evaluation report.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted probabilities
        y_pred_proba (np.ndarray): Predicted probabilities (same as y_pred)
        category_name (str): Name of the category being predicted
        threshold (float): Classification threshold for converting probabilities to binary predictions
        
    Returns:
        Dict[str, float]: Dictionary of evaluation metrics
    """
    # Create reports directory if it doesn't exist
    reports_dir = os.path.join('reports', 'evaluation_reports')
    os.makedirs(reports_dir, exist_ok=True)
    
    # Convert probabilities to binary predictions
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    # Generate plots
    plot_confusion_matrix(y_true, y_pred, category_name, threshold)
    plot_roc_curve(y_true, y_pred_proba, category_name)
    plot_precision_recall_curve(y_true, y_pred_proba, category_name)
    
    # Calculate metrics with zero_division=0
    report = classification_report(y_true, y_pred_binary, output_dict=True, zero_division=0)
    
    # Save report
    save_path = os.path.join(reports_dir, f'evaluation_report_{category_name}.txt')
    with open(save_path, 'w') as f:
        f.write(f"Evaluation Report for {category_name}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Classification threshold: {threshold}\n\n")
        f.write(classification_report(y_true, y_pred_binary, zero_division=0))
        
        # Add additional metrics
        f.write("\nAdditional Metrics:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Number of samples: {len(y_true)}\n")
        f.write(f"Number of positive samples: {sum(y_true)}\n")
        f.write(f"Number of predicted positive samples: {sum(y_pred_binary)}\n")
        f.write(f"Positive class ratio: {sum(y_true)/len(y_true):.2%}\n")
    
    return report

def analyze_feature_importance(model, feature_names: list,
                             save_path: str = None) -> pd.DataFrame:
    """
    Analyze feature importance using model weights.
    
    Args:
        model: Trained neural network model
        feature_names (list): List of feature names
        save_path (str, optional): Path to save the plot
        
    Returns:
        pd.DataFrame: Feature importance scores
    """
    # Get weights from first layer
    weights = np.abs(model.layers[0].get_weights()[0])
    
    # Calculate feature importance
    importance = np.mean(weights, axis=1)
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=importance_df)
    plt.title('Feature Importance')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return importance_df 