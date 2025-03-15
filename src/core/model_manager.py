import os
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from src.core.database import Database
from src.core.feature_engineer import FeatureEngineer
from src.ui.console_ui import ConsoleUI
from src.utils import logger
import tensorflow as tf

class ModelManager:
    def __init__(self, db: Database, feature_engineer: FeatureEngineer):
        self.db = db
        self.feature_engineer = feature_engineer
        self.models = {}  
        self.version = self.db.get_recent_performance().get('model_version', 0)
        self.num_features = len(self.feature_engineer.feature_list)
        self.lookback = self.feature_engineer.lookback_period
        os.makedirs("models", exist_ok=True)
        logger.info(f"Model manager initialized (current version: {self.version})")
        
    def get_model(self, symbol: str, retrain: bool = False) -> tf.keras.Model:
        model_path = f"models/model_{symbol}_v{self.version}.h5"
        if symbol in self.models and not retrain:
            return self.models[symbol]
        if os.path.exists(model_path) and not retrain:
            try:
                self.models[symbol] = tf.keras.models.load_model(model_path)
                logger.info(f"Loaded existing model for {symbol} (v{self.version})")
                return self.models[symbol]
            except Exception as e:
                logger.error(f"Error loading model for {symbol}: {e}")
        self.models[symbol] = self._create_model()
        logger.info(f"Created new model for {symbol}")
        return self.models[symbol]
        
    def _create_model(self) -> tf.keras.Model:
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(self.lookback, self.num_features)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                     loss='mean_squared_error')
        
        return model
    def save_model(self, symbol: str):
        if symbol in self.models:
            model_path = f"models/model_{symbol}_v{self.version}.h5"
            self.models[symbol].save(model_path)
            logger.info(f"Saved model for {symbol} (v{self.version})")
    
    async def train_model(self, symbol: str, df: pd.DataFrame, 
                        sentiment_data: Dict[str, float] = None, 
                        epochs: int = 10, batch_size: int = 32) -> Dict:
        logger.info(f"Training model for {symbol}...")
        
        try:
            X, y = self.feature_engineer.prepare_train_data(symbol, df, sentiment_data)
            if len(X) < batch_size:
                logger.warning(f"Not enough data to train model for {symbol}")
                return {'success': False, 'message': 'Not enough data'}
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            model = self.get_model(symbol, retrain=True)
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=5, restore_best_weights=True
            )
            with ConsoleUI.progress_bar(f"Training {symbol}", range(epochs)) as pbar:
                class ProgressCallback(tf.keras.callbacks.Callback):
                    def on_epoch_end(self, epoch, logs=None):
                        pbar.update(1)
                        pbar.set_postfix({'loss': f"{logs['loss']:.4f}", 'val_loss': f"{logs['val_loss']:.4f}"})
                
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=0,
                    callbacks=[early_stopping, ProgressCallback()]
                )
            mse = model.evaluate(X_val, y_val, verbose=0)
            y_pred = model.predict(X_val)
            mape = mean_absolute_percentage_error(y_val, y_pred) * 100
            direction_actual = np.diff(y_val) > 0
            direction_pred = np.diff(y_pred.flatten()) > 0
            accuracy = np.mean(direction_actual == direction_pred) * 100
            self.save_model(symbol)
            return {
                'success': True,
                'mse': float(mse),
                'mape': float(mape),
                'accuracy': float(accuracy),
                'epochs_completed': len(history.history['loss'])
            }
            
        except Exception as e:
            logger.error(f"Error training model for {symbol}: {e}")
            return {'success': False, 'message': str(e)}
    
    async def predict(self, symbol: str, df: pd.DataFrame, sentiment_score: float = 0.0) -> Dict:
        try:
            X, current_price = self.feature_engineer.prepare_data_for_prediction(symbol, df, sentiment_score)
            model = self.get_model(symbol)
            pred = model.predict(X)[0][0]
            price_change_pct = (pred - current_price) / current_price * 100
            confidence = 50 + min(abs(price_change_pct) * 5, 40) 
            action = 0 if price_change_pct > 1 else 1 if price_change_pct < -1 else 2
            
            return {
                'symbol': symbol,
                'current_price': float(current_price),
                'predicted_price': float(pred),
                'price_change_pct': float(price_change_pct),
                'action': int(action),
                'confidence': float(confidence)
            }
            
        except Exception as e:
            logger.error(f"Error predicting for {symbol}: {e}")
            return {'success': False, 'message': str(e)}
    
    def increment_version(self):
        self.version += 1
        logger.info(f"Incremented model version to {self.version}")
        return self.version
        
    def update_with_reward(self, symbol: str, reward: float):
        logger.info(f"Received reward {reward:.4f} for {symbol}")


