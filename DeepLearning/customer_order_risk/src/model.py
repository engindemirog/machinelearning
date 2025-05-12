import tensorflow as tf
from tensorflow.keras.models  import Sequential,load_model
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
import shap
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from config import MODEL_CONFIG

class ReturnRiskModel:
    def __init__(self):
        self.model = None
        self.history = None

    def build_model(self,input_dim):
        model = Sequential([
            Dense(64,activation="relu",input_shape=(input_dim,)),
            Dropout(0.3),
            Dense(32,activation="relu"),
            Dropout(0.2),
            Dense(16,activation="relu"),
            Dense(1,activation="sigmoid")
        ])

        model.compile(
            optimizer =Adam(),
            loss = "binary_crossentropy",
            metrics = ["accuracy"]
        )

        self.model = model

    def split_data(self,X,y):
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=MODEL_CONFIG["test_size"],random_state=MODEL_CONFIG["random_state"])
        return X_train,X_test,y_train,y_test

    def train(self,X_train,y_train,X_test,y_test):
        

        callbacks = [
           EarlyStopping(monitor="val_loss",patience=5,restore_best_weights=True),
           ModelCheckpoint("best_model.keras",monitor="val_loss",save_best_only=True)
        ]

        self.history = self.model.fit(
            X_train,y_train,
            epochs = MODEL_CONFIG["epochs"],
            validation_data = (X_test,y_test),
            callbacks = callbacks,
            verbose=1
        )

        return X_train,X_test,y_train,y_test

    def evaluate(self,X_test,y_test):
        return self.model.evaluate(X_test,y_test)
    
    def predict(self,X):
        return self.model.predict(X)
    
    def save_model(self,filepath):
        if not filepath:
            print("Wrong file path")
            return
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    def load_model(self,filepath):
        if not filepath:
            print("Wrong file path")
            return
        self.model = load_model(filepath)
        print(f"Model loaded from {filepath}")

    def explain_prediction(self, X, feature_names):
        """
        SHAP değerlerini kullanarak model tahminlerini açıklar.
        
        Args:
            X: Açıklanacak örnekler (numpy array)
            feature_names: Özellik isimleri listesi
            
        Returns:
            DataFrame: Her özelliğin SHAP değerlerini içeren DataFrame
        """
        # SHAP değerlerini hesapla
        explainer = shap.DeepExplainer(self.model, X[:100])  # İlk 100 örneği background olarak kullan
        shap_values = explainer.shap_values(X)
        
        # SHAP değerlerini DataFrame'e dönüştür
        if isinstance(shap_values, list):
            shap_values = shap_values[0]  # Binary classification için ilk değerleri al
            
        shap_df = pd.DataFrame(shap_values, columns=feature_names)
        
        # Her özelliğin mutlak etkisini hesapla
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': np.abs(shap_values).mean(axis=0)
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        return shap_df, feature_importance