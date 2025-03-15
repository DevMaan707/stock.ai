from typing import List
import asyncio
from datetime import datetime
from colorama import Fore, Style
import numpy as np

from src.core.data_manager import DataManager
from src.core.database import Database
from src.core.feature_engineer import FeatureEngineer
from src.core.model_manager import ModelManager
from src.core.recommendation_system import RecommendationEngine
from src.services.news_service import NewsProvider
from src.services.sentiment_service import SentimentAnalyzer
from src.ui.console_ui import ConsoleUI
from src.utils.logger import logger


class StockPredictionSystem:
    def __init__(self, symbols: List[str], db: Database, data_manager: DataManager, 
               feature_engineer: FeatureEngineer, model_manager: ModelManager,
               recommendation_engine: RecommendationEngine, news_provider: NewsProvider,
               sentiment_analyzer: SentimentAnalyzer, ui: ConsoleUI):
        self.symbols = symbols
        self.db = db
        self.data_manager = data_manager
        self.feature_engineer = feature_engineer
        self.model_manager = model_manager
        self.recommendation_engine = recommendation_engine
        self.news_provider = news_provider
        self.sentiment_analyzer = sentiment_analyzer
        self.ui = ui
        
        logger.info(f"Stock prediction system initialized with {len(symbols)} symbols")
        
    async def run_prediction_cycle(self):
        try:
            self.ui.print_header()
            logger.info("Starting prediction cycle...")
            stock_data = await self.data_manager.fetch_data_for_all_symbols()
            news_items = await self.news_provider.get_stock_news(self.symbols)
            
            # Process sentiment directly from AlphaVantage news
            sentiment_by_symbol = {}
            for item in news_items:
                ticker_sentiments = item.get('ticker_sentiments', {})
                for symbol, sentiment_data in ticker_sentiments.items():
                    if symbol in self.symbols:
                        if symbol not in sentiment_by_symbol:
                            sentiment_by_symbol[symbol] = []
                        sentiment_by_symbol[symbol].append(sentiment_data.get('score', 0))
            
            # Average sentiment per symbol
            avg_sentiment = {}
            for symbol, scores in sentiment_by_symbol.items():
                if scores:
                    avg_sentiment[symbol] = np.mean(scores)
                else:
                    avg_sentiment[symbol] = 0.0
            
            for symbol in self.symbols:
                try:
                    if symbol not in stock_data:
                        continue
                    symbol_sentiment = avg_sentiment.get(symbol, 0.0)
                    
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
                    
                    # Save the recommendation to database
                    self.db.save_recommendation(
                        symbol,
                        recommendation['confidence'],
                        recommendation['expected_return'],
                        recommendation['risk_score'],
                        recommendation['recommendation'],
                        recommendation['reasons']
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
                    SELECT predicted_price, current_price, id
                    FROM predictions
                    WHERE symbol = ? AND actual_price IS NULL
                    ORDER BY timestamp DESC LIMIT 1
                ''', (symbol,))
                
                result = cursor.fetchone()
                if not result:
                    continue
                    
                predicted_price, original_price, prediction_id = result
                actual_price = current_prices[symbol]
                predicted_direction = predicted_price > original_price
                actual_direction = actual_price > original_price
                was_correct = predicted_direction == actual_direction
                reward = 1.0 if was_correct else -1.0
                
                self.model_manager.update_with_reward(symbol, reward)
                
                # Update the prediction with actual results
                self.db.conn.execute('''
                    UPDATE predictions
                    SET actual_price = ?, reward = ?, was_correct = ?
                    WHERE id = ?
                ''', (actual_price, reward, 1 if was_correct else 0, prediction_id))
                
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
            
            all_results = []
            for symbol in self.symbols:
                if symbol not in stock_data:
                    continue
                    
                # Get news sentiment for training
                news_items = await self.news_provider.get_news_by_ticker(symbol)
                sentiment_data = {}
                
                # Convert to date-based sentiment dictionary
                for item in news_items:
                    published = item.get('publishedAt', '')
                    if published:
                        try:
                            # Format may vary - handle different formats
                            date_part = published.split('T')[0]
                            sentiment_data[date_part] = item.get('sentiment_score', 0)
                        except Exception:
                            pass
                
                logger.info(f"Training model for {symbol}...")
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
                    all_results.append(results)
                else:
                    logger.error(f"Training failed for {symbol}: {results['message']}")
                    
            # Update database with overall model performance
            if all_results:
                avg_mse = np.mean([r['mse'] for r in all_results])
                avg_mape = np.mean([r['mape'] for r in all_results])
                avg_accuracy = np.mean([r['accuracy'] for r in all_results])
                
                # Increment model version
                new_version = self.model_manager.increment_version()
                
                # Save performance metrics
                self.db.save_model_performance(
                    new_version,
                    avg_mse,
                    avg_mape,
                    avg_accuracy,
                    self.symbols
                )
                    
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
            
    async def run(self):
        """Main run method for the system"""
        while True:
            try:
                await self.run_prediction_cycle()
                await self.validate_predictions()
                
                # Check if it's time for model training
                if datetime.now().hour % 6 == 0 and datetime.now().minute < 30:
                    await self.train_models()
                    
                # Wait for next interval
                logger.info(f"Waiting 30 minutes until next prediction cycle...")
                await asyncio.sleep(30 * 60)
                
            except KeyboardInterrupt:
                logger.info("Shutting down stock prediction system...")
                break
            except Exception as e:
                logger.error(f"Error in stock prediction system: {e}")
                await asyncio.sleep(60)