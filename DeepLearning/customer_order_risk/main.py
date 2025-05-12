from src.database import DatabaseManager
from src.feature_engineering import FeatureEngineer
from src.model import ReturnRiskModel

from sklearn.model_selection import train_test_split
from config import MODEL_CONFIG
import pandas as pd
import numpy as np


def main():
    db_manager = None

    try:
        db_manager = DatabaseManager()
        feature_engineer = FeatureEngineer()
        model = ReturnRiskModel()

        print("Fetching order data")
        df = db_manager.get_order_data()

        print("Creating features")
        df_processed = feature_engineer.create_feautures(df)

        X, y = feature_engineer.prepare_model_data(df_processed)
        feature_names = [
            "unit_price", "quantity", "discount", "total_amount", "discount_amount",
            "avg_order_amount", "std_order_amount", "total_spent", "avg_discount",
            "max_discount", "avg_quantity", "total_quantity"
        ]

        X_train, X_test, y_train, y_test = model.split_data(X, y)
        
        model.build_model(input_dim=X_train.shape[1])
        model.train(X_train, y_train, X_test, y_test)
        loss, accuracy = model.evaluate(X_test, y_test)

        print(f"Test accuracy: {accuracy}")

        # Riskli bulunan siparişleri belirle
        predictions = model.predict(X_test)
        risky_orders = X_test[predictions.flatten() > 0.5]  # 0.5'ten büyük tahminleri riskli kabul et

        if len(risky_orders) > 0:
            print("\nRiskli bulunan siparişlerin açıklaması:")
            shap_df, feature_importance = model.explain_prediction(risky_orders, feature_names)
            
            print("\nEn önemli özellikler (SHAP değerlerine göre):")
            print(feature_importance.head())
            
            print("\nİlk riskli sipariş için özellik katkıları:")
            first_risky = shap_df.iloc[0]
            for feature, value in first_risky.items():
                if abs(value) > 0.01:  # Sadece önemli katkıları göster
                    direction = "arttırdı" if value > 0 else "azalttı"
                    print(f"{feature}: {value:.4f} ({direction})")

    except Exception as e:
        print(e)
    finally:
        if db_manager is not None:
            db_manager.disconnect()

if __name__ == "__main__":
    main()