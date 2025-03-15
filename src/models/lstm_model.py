import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np

class LSTMModel:
    """Long Short-Term Memory model for time series prediction"""
    
    def __init__(self, lookback_period=30, feature_count=16, lstm_units=64):
        """
        Initialize the LSTM model
        
        Args:
            lookback_period: Number of time steps to look back
            feature_count: Number of features in the input data
            lstm_units: Number of LSTM units in the first layer
        """
        self.lookback_period = lookback_period
        self.feature_count = feature_count
        self.lstm_units = lstm_units
        self.model = self._build_model()
        
    def _build_model(self):
        """Build and compile the LSTM model"""
        model = Sequential([
            LSTM(self.lstm_units, return_sequences=True, 
                 input_shape=(self.lookback_period, self.feature_count)),
            Dropout(0.2),
            LSTM(self.lstm_units // 2),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='mean_squared_error')
        
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=50, batch_size=32, early_stopping=True, 
              callbacks=None):
        """
        Train the LSTM model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Training batch size
            early_stopping: Whether to use early stopping
            callbacks: Additional callbacks for training
            
        Returns:
            Training history
        """
        all_callbacks = callbacks or []
        
        if early_stopping and X_val is not None and y_val is not None:
            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=5, restore_best_weights=True
            )
            all_callbacks.append(early_stop)
        
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=all_callbacks
        )
        
        return history
    
    def predict(self, X):
        """Generate predictions"""
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        """Evaluate model performance"""
        return self.model.evaluate(X, y, verbose=0)
    
    def save(self, path):
        """Save the model to disk"""
        self.model.save(path)
    
    @classmethod
    def load(cls, path):
        """Load a model from disk"""
        instance = cls(1, 1, 1)  # Create a dummy instance
        instance.model = tf.keras.models.load_model(path)
        return instance
    
    def summary(self):
        """Get model summary"""
        return self.model.summary()