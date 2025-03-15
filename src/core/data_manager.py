import asyncio
from datetime import datetime, timedelta
from typing import Dict, List
import yfinance as yf
import pandas as pd

from src.utils import logger


class DataManager:
    def __init__(self, symbols: List[str], lookback_days: int = 500):
        self.symbols = symbols
        self.lookback_days = lookback_days
        self.data_cache = {}  
        self.cache_timestamp = {}  
        logger.info(f"Data manager initialized for {len(symbols)} symbols")
        
    async def fetch_stock_data(self, symbol: str, force_refresh: bool = False) -> pd.DataFrame:
        current_time = datetime.now()
        if (not force_refresh and 
            symbol in self.data_cache and 
            symbol in self.cache_timestamp and
            (current_time - self.cache_timestamp[symbol]).total_seconds() < 3600):  # Cache for 1 hour
            return self.data_cache[symbol]
            
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.lookback_days)
            
            logger.info(f"Fetching data for {symbol} from {start_date.date()} to {end_date.date()}")
            stock_data = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                progress=False
            )
            
            if stock_data.empty:
                logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()
            self.data_cache[symbol] = stock_data
            self.cache_timestamp[symbol] = current_time
            
            logger.info(f"Fetched {len(stock_data)} data points for {symbol}")
            return stock_data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    async def get_latest_price(self, symbol: str) -> float:
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period="1d")
            
            if data.empty:
                logger.warning(f"No current price data for {symbol}")
                return None
                
            return data['Close'].iloc[-1]
            
        except Exception as e:
            logger.error(f"Error getting latest price for {symbol}: {e}")
            return None
    
    async def fetch_data_for_all_symbols(self, force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
        results = {}
        async with asyncio.TaskGroup() as tg:
            tasks = {
                symbol: tg.create_task(self.fetch_stock_data(symbol, force_refresh))
                for symbol in self.symbols
            }
        for symbol, task in tasks.items():
            df = task.result()
            if not df.empty:
                results[symbol] = df
                
        return results

