"""
Main script for customer category prediction.
"""
import os
import logging
from src.data.database import get_customer_order_history
from src.data.feature_engineering import (
    create_customer_features,
    create_category_features,
    create_time_based_features,
    prepare_model_data
)
from src.models.neural_network import CustomerCategoryPredictor
from src.models.model_evaluation import (
    plot_training_history,
    plot_roc_curve,
    plot_confusion_matrix,
    generate_evaluation_report
)
from src.config import MODEL_CONFIG, FEATURE_CONFIG

def main():
    """
    Main function to run the customer category prediction model.
    """
    # Get raw data
    raw_data = get_customer_order_history()
    print("Raw data shape:", raw_data.shape)
    print("Raw data columns:", raw_data.columns.tolist())
    
    # Create features
    customer_features = create_customer_features(raw_data)
    print("\nCustomer features shape:", customer_features.shape)
    print("Customer features columns:", customer_features.columns.tolist())
    
    category_features = create_category_features(raw_data)
    print("\nCategory features shape:", category_features.shape)
    print("Category features columns:", category_features.columns.tolist())
    
    time_features = create_time_based_features(raw_data)
    print("\nTime features shape:", time_features.shape)
    print("Time features columns:", time_features.columns.tolist())
    
    # Merge features
    # Start with unique customer-category pairs
    df = raw_data[['customer_id', 'category_id']].drop_duplicates()
    
    # Merge customer features
    df = df.merge(customer_features, on='customer_id', how='left')
    
    # Merge category features
    df = df.merge(category_features, on=['customer_id', 'category_id'], how='left')
    
    # Merge time features (using only the latest time features for each customer)
    latest_time_features = time_features.sort_values('order_date').groupby('customer_id').last()
    df = df.merge(latest_time_features, on='customer_id', how='left')
    
    # Drop duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]
    
    print("\nFinal features shape:", df.shape)
    print("Final features columns:", df.columns.tolist())
    
    # Prepare model data for each target category
    for target_category in FEATURE_CONFIG['target_categories']:
        print(f"\nTraining model for category {target_category}")
        
        # Prepare data
        X_train, X_test, y_train, y_test = prepare_model_data(df, target_category)
        
        # Initialize and train model
        model = CustomerCategoryPredictor(
            input_dim=X_train.shape[1],
            hidden_layers=MODEL_CONFIG['hidden_layers'],
            dropout_rate=MODEL_CONFIG['dropout_rate'],
            learning_rate=MODEL_CONFIG['learning_rate']
        )
        
        # Train model
        history = model.train(
            X_train, y_train,
            batch_size=MODEL_CONFIG['batch_size'],
            epochs=MODEL_CONFIG['epochs'],
            validation_split=MODEL_CONFIG['validation_split']
        )
        
        # Plot training history
        plot_training_history(history, f"Category_{target_category}")
        
        # Make predictions on test set
        y_pred = model.predict(X_test)
        
        # Plot ROC curve
        plot_roc_curve(y_test, y_pred, f"Category_{target_category}")
        
        # Plot confusion matrix
        plot_confusion_matrix(y_test, y_pred, f"Category_{target_category}")
        
        # Generate evaluation report
        generate_evaluation_report(
            y_true=y_test,
            y_pred=y_pred,
            y_pred_proba=y_pred,
            category_name=f"Category_{target_category}",
            threshold=0.5
        )
        
        # Save model
        model_path = os.path.join('models', 'saved', f'category_{target_category}_model.keras')
        model.save(model_path)
        print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main() 