import os
import pickle
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from src.core.database import Database
from src.utils import logger
import talib

class FeatureEngineer:
    
    def __init__(self, db: Database):
        self.db = db
        self.scalers = {} 
        self.lookback_period = 30  
        self.feature_list = [
            'Close', 'Volume', 'SMA_5', 'SMA_20', 'EMA_12', 'EMA_26', 
            'RSI', 'MACD', 'MACD_Signal', 'Upper_Band', 'Lower_Band',
            'Stoch_K', 'Stoch_D', 'ATR', 'ADX', 'Sentiment'
        ]
        logger.info("Feature engineer initialized")
        
    def get_scaler(self, symbol: str) -> MinMaxScaler:
        if symbol not in self.scalers:
            scaler_path = f"models/scaler_{symbol}.pkl"
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scalers[symbol] = pickle.load(f)
            else:
                self.scalers[symbol] = MinMaxScaler(feature_range=(0, 1))
        return self.scalers[symbol]
        
    def save_scaler(self, symbol: str):
        if symbol in self.scalers:
            os.makedirs("models", exist_ok=True)
            with open(f"models/scaler_{symbol}.pkl", 'wb') as f:
                pickle.dump(self.scalers[symbol], f)
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) < 30:  
            logger.warning(f"Not enough data for technical indicators: {len(df)} rows")
            return df

        df['SMA_5'] = talib.SMA(df['Close'].values, timeperiod=5)
        df['SMA_20'] = talib.SMA(df['Close'].values, timeperiod=20)
        df['EMA_12'] = talib.EMA(df['Close'].values, timeperiod=12)
        df['EMA_26'] = talib.EMA(df['Close'].values, timeperiod=26)
        df['RSI'] = talib.RSI(df['Close'].values, timeperiod=14)
        
        macd, signal, _ = talib.MACD(df['Close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
        df['MACD'] = macd
        df['MACD_Signal'] = signal
        
        upper, middle, lower = talib.BBANDS(df['Close'].values, timeperiod=20)
        df['Upper_Band'] = upper
        df['Lower_Band'] = lower

        stoch_k, stoch_d = talib.STOCH(df['High'].values, df['Low'].values, df['Close'].values, 
                                      fastk_period=14, slowk_period=3, slowd_period=3)
        df['Stoch_K'] = stoch_k
        df['Stoch_D'] = stoch_d
        
        df['ATR'] = talib.ATR(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=14)
        df['ADX'] = talib.ADX(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=14)

        df['Sentiment'] = 0.0
    
        df.fillna(0, inplace=True)
        
        return df
        
    def prepare_data_for_prediction(self, symbol: str, df: pd.DataFrame, 
                                  sentiment_score: float = 0.0) -> Tuple[np.ndarray, float]:
        if len(df) < self.lookback_period:
            raise ValueError(f"Not enough data for {symbol}, needed {self.lookback_period} days")
        df = self.add_technical_indicators(df)
        df.loc[df.index[-1], 'Sentiment'] = sentiment_score
        feature_data = df[self.feature_list].values
        scaler = self.get_scaler(symbol)
        if len(feature_data) > self.lookback_period * 2:
            scaler.fit(feature_data[-self.lookback_period * 2:])
        else:
            scaler.fit(feature_data)
        scaled_data = scaler.transform(feature_data)
        self.save_scaler(symbol)
        current_price = df['Close'].iloc[-1]
        X = np.array([scaled_data[-self.lookback_period:]])
        
        return X, current_price
        
    def prepare_train_data(self, symbol: str, df: pd.DataFrame, 
                         sentiment_data: Dict[str, float] = None) -> Tuple[np.ndarray, np.ndarray]:
        df = self.add_technical_indicators(df)

        if sentiment_data:
            for date_str, score in sentiment_data.items():
                try:
                    date = pd.to_datetime(date_str)
                    if date in df.index:
                        df.loc[date, 'Sentiment'] = score
                except Exception as e:
                    logger.warning(f"Error adding sentiment for {date_str}: {e}")
        feature_data = df[self.feature_list].values
        target_data = df['Close'].shift(-1).values[:-1]
        feature_data = feature_data[:-1] 
        scaler = self.get_scaler(symbol)
        scaler.fit(feature_data)
        scaled_features = scaler.transform(feature_data)
        self.save_scaler(symbol)
        X, y = [], []
        for i in range(len(scaled_features) - self.lookback_period):
            X.append(scaled_features[i:i + self.lookback_period])
            y.append(target_data[i + self.lookback_period])
            
        return np.array(X), np.array(y)
