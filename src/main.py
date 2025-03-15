import asyncio
import argparse
import os
import time
from datetime import datetime
import colorama
from dotenv import load_dotenv

from src.core.data_manager import DataManager
from src.core.database import Database
from src.core.feature_engineer import FeatureEngineer
from src.core.model_manager import ModelManager
from src.core.recommendation_system import RecommendationEngine
from src.core.stock_prediction_system import StockPredictionSystem
from src.services.news_service import NewsProvider
from src.services.sentiment_service import SentimentAnalyzer
from src.ui.console_ui import ConsoleUI
from src.utils.logger import logger
from src.utils.config import Config

# Initialize colorama for cross-platform colored terminal output
colorama.init()

# Load environment variables
load_dotenv()

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Stock Market Prediction System')
    parser.add_argument('--symbols', type=str, nargs='+', 
                      default=Config.DEFAULT_SYMBOLS,
                      help=f'List of stock symbols to monitor (default: {Config.DEFAULT_SYMBOLS})')
    parser.add_argument('--interval', type=int, default=Config.DEFAULT_INTERVAL,
                      help=f'Prediction interval in minutes (default: {Config.DEFAULT_INTERVAL})')
    parser.add_argument('--train', action='store_true',
                      help='Force training of models on startup')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug logging')
    return parser.parse_args()

class StockAI:
    """Main application class that initializes and runs the stock prediction system"""
    
    def __init__(self, symbols, interval, train_on_start=False):
        """Initialize the StockAI application"""
        logger.info(f"Initializing StockAI with symbols: {symbols}, interval: {interval} minutes")
        self.symbols = symbols
        self.interval = interval
        self.train_on_start = train_on_start
        
        # Initialize components
        self.db = Database(Config.DATABASE_PATH)
        self.data_manager = DataManager(symbols, Config.LOOKBACK_DAYS)
        self.feature_engineer = FeatureEngineer(self.db)
        self.model_manager = ModelManager(self.db, self.feature_engineer)
        self.news_provider = NewsProvider()
        self.sentiment_analyzer = SentimentAnalyzer(Config.LLM_ENDPOINT)
        self.recommendation_engine = RecommendationEngine(
            self.db, self.model_manager, self.data_manager
        )
        self.ui = ConsoleUI()
        
        # Create the stock prediction system
        self.system = StockPredictionSystem(
            symbols=symbols,
            db=self.db,
            data_manager=self.data_manager, 
            feature_engineer=self.feature_engineer,
            model_manager=self.model_manager,
            recommendation_engine=self.recommendation_engine,
            news_provider=self.news_provider,
            sentiment_analyzer=self.sentiment_analyzer,
            ui=self.ui
        )
    
    async def run(self):
        """Run the StockAI application"""
        try:
            self.ui.print_header()
            
            # Initialize external services
            await self.data_manager.initialize()
            await self.news_provider.initialize()
            await self.sentiment_analyzer.initialize()
            
            # Force training if requested
            if self.train_on_start:
                logger.info("Initial model training requested")
                await self.system.train_models()
            
            # Main application loop
            while True:
                try:
                    # Run prediction cycle
                    await self.system.run_prediction_cycle()
                    
                    # Validate previous predictions
                    await self.system.validate_predictions()
                    
                    # Check if it's time for periodic training (every 6 hours)
                    current_hour = datetime.now().hour
                    if current_hour % 6 == 0 and datetime.now().minute < self.interval:
                        logger.info("Scheduled model training starting")
                        await self.system.train_models()
                    
                    # Wait for next interval
                    logger.info(f"Waiting {self.interval} minutes until next prediction cycle...")
                    await asyncio.sleep(self.interval * 60)
                    
                except KeyboardInterrupt:
                    logger.info("Keyboard interrupt received, shutting down...")
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    await asyncio.sleep(60)  # Wait before retrying
        
        finally:
            # Cleanup resources
            await self.data_manager.close()
            await self.news_provider.close()
            await self.sentiment_analyzer.close()
            logger.info("StockAI shutdown complete")

async def main():
    """Application entry point"""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Configure logging level
    if args.debug:
        import logging
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Create and run the application
    app = StockAI(
        symbols=args.symbols,
        interval=args.interval,
        train_on_start=args.train
    )
    
    await app.run()

if __name__ == "__main__":
    # Run the main async function
    asyncio.run(main())