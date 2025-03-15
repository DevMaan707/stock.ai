from datetime import datetime
import sqlite3
import os
import threading
from src.utils import logger
from typing import Dict, List, Tuple, Union, Any
class Database:
    def __init__(self, db_path: str = 'data/stock_predictor.db'):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.lock = threading.Lock()
        self.create_tables()
        logger.info("Database initialized")

    def create_tables(self):
        with self.conn:
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    timestamp DATETIME,
                    action INTEGER,
                    confidence REAL,
                    current_price REAL,
                    predicted_price REAL,
                    actual_price REAL,
                    reward REAL,
                    was_correct INTEGER,
                    model_version INTEGER
                )
            ''')
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_version INTEGER,
                    timestamp DATETIME,
                    mse REAL,
                    mape REAL,
                    accuracy REAL,
                    symbols TEXT
                )
            ''')
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS stock_recommendations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    symbol TEXT,
                    confidence REAL,
                    expected_return REAL,
                    risk_score REAL,
                    recommendation TEXT,
                    reasons TEXT
                )
            ''')
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS news_sentiment (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    symbol TEXT,
                    news_title TEXT,
                    news_snippet TEXT,
                    sentiment_score REAL
                )
            ''')

    def save_prediction(self, symbol: str, action: int, confidence: float,
                       current_price: float, predicted_price: float, 
                       model_version: int):
        with self.lock:
            with self.conn:
                self.conn.execute('''
                    INSERT INTO predictions 
                    (symbol, timestamp, action, confidence, current_price, 
                     predicted_price, model_version)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (symbol, datetime.now(), action, confidence, 
                     current_price, predicted_price, model_version))

    def update_prediction_result(self, symbol: str, actual_price: float, 
                               reward: float, was_correct: bool):
        with self.lock:
            with self.conn:
                cursor = self.conn.execute('''
                    SELECT id FROM predictions 
                    WHERE symbol = ? AND actual_price IS NULL
                    ORDER BY timestamp DESC LIMIT 1
                ''', (symbol,))
                
                result = cursor.fetchone()
                if result:
                    prediction_id = result[0]
                    self.conn.execute('''
                        UPDATE predictions 
                        SET actual_price = ?, reward = ?, was_correct = ?
                        WHERE id = ?
                    ''', (actual_price, reward, 1 if was_correct else 0, prediction_id))

    def save_model_performance(self, model_version: int, mse: float, 
                             mape: float, accuracy: float, symbols: List[str]):
        with self.lock:
            with self.conn:
                self.conn.execute('''
                    INSERT INTO model_performance 
                    (model_version, timestamp, mse, mape, accuracy, symbols)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (model_version, datetime.now(), mse, mape, accuracy, 
                     ','.join(symbols)))

    def save_recommendation(self, symbol: str, confidence: float, 
                          expected_return: float, risk_score: float,
                          recommendation: str, reasons: str):
        with self.lock:
            with self.conn:
                self.conn.execute('''
                    INSERT INTO stock_recommendations 
                    (timestamp, symbol, confidence, expected_return, 
                     risk_score, recommendation, reasons)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (datetime.now(), symbol, confidence, expected_return, 
                     risk_score, recommendation, reasons))

    def save_news_sentiment(self, symbol: str, news_title: str, 
                          news_snippet: str, sentiment_score: float):
        with self.lock:
            with self.conn:
                self.conn.execute('''
                    INSERT INTO news_sentiment 
                    (timestamp, symbol, news_title, news_snippet, sentiment_score)
                    VALUES (?, ?, ?, ?, ?)
                ''', (datetime.now(), symbol, news_title, news_snippet, sentiment_score))

    def get_recent_performance(self) -> Dict:
        with self.lock:
            cursor = self.conn.execute('''
                SELECT model_version, mse, mape, accuracy 
                FROM model_performance 
                ORDER BY timestamp DESC LIMIT 1
            ''')
            
            result = cursor.fetchone()
            if result:
                return {
                    'model_version': result[0],
                    'mse': result[1],
                    'mape': result[2],
                    'accuracy': result[3]
                }
            return {'model_version': 0, 'mse': 0, 'mape': 0, 'accuracy': 0}

    def get_prediction_accuracy(self, symbol: str = None, 
                              timeframe: str = "all") -> float:
        with self.lock:
            query = '''
                SELECT COUNT(*) FROM predictions 
                WHERE actual_price IS NOT NULL
            '''
            params = []
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
                
            if timeframe == "day":
                query += " AND timestamp > datetime('now', '-1 day')"
            elif timeframe == "week":
                query += " AND timestamp > datetime('now', '-7 days')"
            elif timeframe == "month":
                query += " AND timestamp > datetime('now', '-30 days')"
                
            cursor = self.conn.execute(query, params)
            total = cursor.fetchone()[0]
            
            if total == 0:
                return 0.0
                
            query = '''
                SELECT COUNT(*) FROM predictions 
                WHERE actual_price IS NOT NULL AND was_correct = 1
            '''
            
            if symbol:
                query += " AND symbol = ?"
                
            if timeframe == "day":
                query += " AND timestamp > datetime('now', '-1 day')"
            elif timeframe == "week":
                query += " AND timestamp > datetime('now', '-7 days')"
            elif timeframe == "month":
                query += " AND timestamp > datetime('now', '-30 days')"
                
            cursor = self.conn.execute(query, params)
            correct = cursor.fetchone()[0]
            
            return (correct / total) * 100 if total > 0 else 0

    def get_top_recommended_stocks(self, limit: int = 5) -> List[Dict]:
        with self.lock:
            cursor = self.conn.execute('''
                SELECT symbol, confidence, expected_return, risk_score, recommendation 
                FROM stock_recommendations 
                WHERE timestamp > datetime('now', '-1 day')
                ORDER BY expected_return DESC, confidence DESC
                LIMIT ?
            ''', (limit,))
            
            results = cursor.fetchall()
            return [
                {
                    'symbol': r[0],
                    'confidence': r[1],
                    'expected_return': r[2],
                    'risk_score': r[3],
                    'recommendation': r[4]
                }
                for r in results
            ]
