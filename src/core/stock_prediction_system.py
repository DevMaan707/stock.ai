from typing import List
from colorama import Fore, Style
import numpy as np

from src.core.data_manager import DataManager
from src.core.database import Database
from src.core.feature_engineer import FeatureEngineer
from src.core.model_manager import ModelManager
from src.services.news_service import NewsProvider
from src.services.sentiment_service import SentimentAnalyzer
from src.ui.console_ui import ConsoleUI
from src.utils import logger
from src.core.recommendation_system import RecommendationEngine

class StockPredictionSystem:
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.db = Database()
        self.data_manager = DataManager(symbols)
        self.feature_engineer = FeatureEngineer(self.db)
        self.model_manager = ModelManager(self.db, self.feature_engineer)
        self.recommendation_engine = RecommendationEngine(self.db, self.model_manager, self.data_manager)
        self.news_provider = NewsProvider()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.ui = ConsoleUI()
        
        logger.info(f"Stock prediction system initialized with {len(symbols)} symbols")
        
    async def run_prediction_cycle(self):
        try:
            self.ui.print_header()
            logger.info("Starting prediction cycle...")
            stock_data = await self.data_manager.fetch_data_for_all_symbols()
            news_items = await self.news_provider.get_stock_news(self.symbols)
            sentiment_results = await self.sentiment_analyzer.analyze_news_batch(news_items)
            for symbol in self.symbols:
                try:
                    if symbol not in stock_data:
                        continue
                    symbol_sentiment = np.mean([
                        item['sentiment'] for item in sentiment_results
                        if item['symbol'] == symbol
                    ]) if sentiment_results else 0.0
                    prediction = await self.model_manager.predict(
                        symbol, stock_data[symbol], symbol_sentiment
                    )
                    
                    if 'success' in prediction and not prediction['success']:
                        continue
                    recommendation = await self.recommendation_engine.analyze_stock(
                        symbol, prediction, symbol_sentiment
                    )
                    self.ui.print_prediction(
                        symbol,
                        prediction['action'],
                        prediction['confidence'],
                        prediction['current_price'],
                        prediction['predicted_price']
                    )
                    self.db.save_prediction(
                        symbol,
                        prediction['action'],
                        prediction['confidence'],
                        prediction['current_price'],
                        prediction['predicted_price'],
                        self.model_manager.version
                    )
                    
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
                    continue
            self._display_system_stats()
            
        except Exception as e:
            logger.error(f"Error in prediction cycle: {e}")
    
    async def validate_predictions(self):
        try:
            logger.info("Validating previous predictions...")
            current_prices = {}
            for symbol in self.symbols:
                price = await self.data_manager.get_latest_price(symbol)
                if price is not None:
                    current_prices[symbol] = price
            for symbol in self.symbols:
                if symbol not in current_prices:
                    continue
                cursor = self.db.conn.execute('''
                    SELECT predicted_price, current_price
                    FROM predictions
                    WHERE symbol = ? AND actual_price IS NULL
                    ORDER BY timestamp DESC LIMIT 1
                ''', (symbol,))
                
                result = cursor.fetchone()
                if not result:
                    continue
                predicted_price, original_price = result
                actual_price = current_prices[symbol]
                predicted_direction = predicted_price > original_price
                actual_direction = actual_price > original_price
                was_correct = predicted_direction == actual_direction
                if was_correct:
                    reward = 1.0
                else:
                    reward = -1.0
                self.model_manager.update_with_reward(symbol, reward)
                self.ui.print_validation_result(
                    symbol,
                    "UP" if predicted_direction else "DOWN",
                    "UP" if actual_direction else "DOWN",
                    reward
                )
                
        except Exception as e:
            logger.error(f"Error in prediction validation: {e}")
    
    async def train_models(self):
        try:
            logger.info("Starting model training cycle...")
            stock_data = await self.data_manager.fetch_data_for_all_symbols(force_refresh=True)
            for symbol in self.symbols:
                if symbol not in stock_data:
                    continue
                    
                logger.info(f"Training model for {symbol}...")
                sentiment_data = {}  
                results = await self.model_manager.train_model(
                    symbol,
                    stock_data[symbol],
                    sentiment_data
                )
                
                if results['success']:
                    logger.info(f"Training completed for {symbol}: "
                              f"MSE={results['mse']:.4f}, "
                              f"MAPE={results['mape']:.2f}%, "
                              f"Accuracy={results['accuracy']:.2f}%")
                else:
                    logger.error(f"Training failed for {symbol}: {results['message']}")
                    
        except Exception as e:
            logger.error(f"Error in model training: {e}")
    
    def _display_system_stats(self):
        try:
            stats = self.db.get_recent_performance()
            print(f"\n{Fore.CYAN}System Statistics:{Style.RESET_ALL}")
            print(f"Model Version: {stats['model_version']}")
            print(f"Overall Accuracy: {stats['accuracy']:.2f}%")
            print(f"MSE: {stats['mse']:.4f}")
            print(f"MAPE: {stats['mape']:.2f}%")
            
        except Exception as e:
            logger.error(f"Error displaying system stats: {e}")
