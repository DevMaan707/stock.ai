import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    NEWS_API_KEY = os.getenv('NEWS_API_KEY')
    LLM_ENDPOINT = os.getenv('LLM_ENDPOINT', 'http://localhost:8080/generate')
    DATABASE_PATH = os.getenv('DATABASE_PATH', 'data/database/stock_predictor.db')
    MODEL_PATH = os.getenv('MODEL_PATH', 'models/trained/')
    LOG_PATH = os.getenv('LOG_PATH', 'logs/')
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    DEFAULT_SYMBOLS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    DEFAULT_INTERVAL = 60
    LOOKBACK_DAYS = 500
    TRAINING_EPOCHS = 10
    BATCH_SIZE = 32
    TECHNICAL_INDICATORS = [
        'RSI', 'MACD', 'ATR', 'BBANDS', 'OBV'
    ]