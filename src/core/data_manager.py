import asyncio
from datetime import datetime, timedelta
from typing import Dict, List
import aiohttp
import numpy as np
import pandas as pd
import os
import json
import time
import random

from src.utils.logger import logger
from src.utils.config import Config
from src.utils.helpers import ensure_directory, save_to_json, load_from_json

class DataManager:
    def __init__(self, symbols: List[str], lookback_days: int = 500):
        self.symbols = symbols
        self.lookback_days = lookback_days
        self.data_cache = {}  
        self.cache_timestamp = {}
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY', 'MS0Q3VIDCHALT4MT')
        self.session = None
        self.last_api_call = 0
        self.min_call_interval = 15  # seconds between API calls
        ensure_directory('data/raw')
        logger.info(f"Data manager initialized for {len(symbols)} symbols")
        
    async def initialize(self):
        """Initialize the HTTP session"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
    
    async def close(self):
        """Close the HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def _rate_limit(self):
        """Implement rate limiting for API calls"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call
        
        if time_since_last_call < self.min_call_interval:
            wait_time = self.min_call_interval - time_since_last_call + random.uniform(0.1, 1.0)
            logger.info(f"Rate limiting: waiting {wait_time:.2f} seconds")
            await asyncio.sleep(wait_time)
            
        self.last_api_call = time.time()
            
    async def fetch_stock_data(self, symbol: str, force_refresh: bool = False) -> pd.DataFrame:
        # First, check cache
        current_time = datetime.now()
        if (not force_refresh and 
            symbol in self.data_cache and 
            symbol in self.cache_timestamp and
            (current_time - self.cache_timestamp[symbol]).total_seconds() < 3600):  # Cache for 1 hour
            return self.data_cache[symbol]
        
        # Then check local storage
        cache_file = f"data/raw/{symbol}_daily.json"
        if not force_refresh and os.path.exists(cache_file):
            file_age = time.time() - os.path.getmtime(cache_file)
            if file_age < 86400:  # Less than 24 hours old
                try:
                    data = load_from_json(cache_file)
                    if data and 'Time Series (Daily)' in data:
                        logger.info(f"Loading {symbol} data from local cache")
                        return self._process_alpha_vantage_data(data, symbol, current_time)
                except Exception as e:
                    logger.error(f"Error loading cached data for {symbol}: {e}")
        
        # If we need to fetch from API
        try:
            if not self.session:
                await self.initialize()
            
            # Apply rate limiting
            await self._rate_limit()
                
            # Use AlphaVantage API
            url = (f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED"
                  f"&symbol={symbol}&outputsize=full&apikey={self.alpha_vantage_key}")
            
            logger.info(f"Fetching data for {symbol} from AlphaVantage")
            
            async with self.session.get(url) as response:
                if response.status != 200:
                    logger.error(f"Error fetching data for {symbol}: Status {response.status}")
                    return self._get_mock_data(symbol)
                
                data = await response.json()
                
                # Save response to local cache regardless of content
                save_to_json(data, cache_file)
                
                if 'Error Message' in data:
                    logger.error(f"AlphaVantage API error for {symbol}: {data['Error Message']}")
                    return self._get_mock_data(symbol)
                
                if 'Information' in data:
                    logger.warning(f"AlphaVantage API limit hit: {data['Information']}")
                    # Try to use cached data even if it's older
                    if os.path.exists(cache_file):
                        try:
                            data = load_from_json(cache_file)
                            if data and 'Time Series (Daily)' in data:
                                logger.info(f"Using older cached data for {symbol}")
                                return self._process_alpha_vantage_data(data, symbol, current_time)
                        except Exception as e:
                            logger.error(f"Error loading cached data for {symbol}: {e}")
                    return self._get_mock_data(symbol)
                
                if 'Time Series (Daily)' not in data:
                    logger.error(f"No time series data for {symbol}: {data.keys()}")
                    return self._get_mock_data(symbol)
                
                return self._process_alpha_vantage_data(data, symbol, current_time)
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return self._get_mock_data(symbol)
    
    def _process_alpha_vantage_data(self, data, symbol, current_time):
        """Process the AlphaVantage JSON data into a DataFrame"""
        time_series = data['Time Series (Daily)']
        records = []
        
        for date, values in time_series.items():
            records.append({
                'Date': date,
                'Open': float(values['1. open']),
                'High': float(values['2. high']),
                'Low': float(values['3. low']),
                'Close': float(values['4. close']),
                'Adjusted Close': float(values['5. adjusted close']),
                'Volume': int(values['6. volume']),
            })
        
        df = pd.DataFrame(records)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        
        # Limit to lookback period
        start_date = current_time - timedelta(days=self.lookback_days)
        df = df[df.index >= start_date]
        
        self.data_cache[symbol] = df
        self.cache_timestamp[symbol] = current_time
        
        logger.info(f"Processed {len(df)} data points for {symbol}")
        return df
    
    def _get_mock_data(self, symbol):
        """Generate mock data when API fails - for demonstration only"""
        logger.warning(f"Generating mock data for {symbol}")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days)
        
        # Generate dates
        dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
        
        # Generate random prices with a trend
        base_price = 100.0 if symbol in ['AAPL', 'MSFT'] else 50.0
        trend = 0.001  # Small upward trend
        volatility = 0.02  # 2% daily volatility
        
        np.random.seed(hash(symbol) % 10000)  # Different seed for each symbol
        
        # Generate log returns with drift
        log_returns = np.random.normal(trend, volatility, size=len(dates))
        # Convert to price series
        prices = base_price * np.exp(np.cumsum(log_returns))
        
        # Create DataFrame
        df = pd.DataFrame({
            'Open': prices * (1 - volatility/2),
            'High': prices * (1 + volatility),
            'Low': prices * (1 - volatility),
            'Close': prices,
            'Adjusted Close': prices,
            'Volume': np.random.randint(1000000, 10000000, size=len(dates))
        }, index=dates)
        
        self.data_cache[symbol] = df
        self.cache_timestamp[symbol] = datetime.now()
        
        return df
    
    async def get_latest_price(self, symbol: str) -> float:
        try:
            # First check if we have recent data in cache
            if symbol in self.data_cache:
                df = self.data_cache[symbol]
                if not df.empty:
                    return df['Close'].iloc[-1]
            
            # If we need to fetch from API
            if not self.session:
                await self.initialize()
                
            # Apply rate limiting
            await self._rate_limit()
                
            # Use AlphaVantage API for latest quote
            url = (f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE"
                  f"&symbol={symbol}&apikey={self.alpha_vantage_key}")
            
            async with self.session.get(url) as response:
                if response.status != 200:
                    logger.warning(f"Error getting latest price for {symbol}: Status {response.status}")
                    return self._get_mock_price(symbol)
                
                data = await response.json()
                
                if 'Information' in data:
                    logger.warning(f"AlphaVantage API limit hit when getting price: {data['Information']}")
                    return self._get_mock_price(symbol)
                
                if 'Global Quote' not in data:
                    logger.warning(f"No quote data for {symbol}")
                    return self._get_mock_price(symbol)
                    
                quote = data['Global Quote']
                if '05. price' in quote:
                    price = float(quote['05. price'])
                    return price
                    
                return self._get_mock_price(symbol)
                
        except Exception as e:
            logger.error(f"Error getting latest price for {symbol}: {e}")
            return self._get_mock_price(symbol)
    
    def _get_mock_price(self, symbol):
        """Generate a mock price when API fails"""
        if symbol in self.data_cache:
            df = self.data_cache[symbol]
            if not df.empty:
                # Use the last price with a small random change
                last_price = df['Close'].iloc[-1]
                change = last_price * np.random.uniform(-0.01, 0.01)  # ±1% change
                return last_price + change
        
        # Fallback base prices
        base_prices = {
            'AAPL': 175.0,
            'GOOGL': 140.0,
            'MSFT': 380.0,
            'AMZN': 175.0
        }
        
        return base_prices.get(symbol, 100.0) * (1 + np.random.uniform(-0.01, 0.01))
    
    async def fetch_data_for_all_symbols(self, force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
        if not self.session:
            await self.initialize()
            
        results = {}
        
        # For AlphaVantage's strict rate limits, fetch one symbol at a time with delays
        for symbol in self.symbols:
            df = await self.fetch_stock_data(symbol, force_refresh)
            if not df.empty:
                results[symbol] = df
                
        return results
        
    async def fetch_company_news(self, symbol: str) -> List[Dict]:
        """Fetch news for a specific company using AlphaVantage"""
        try:
            if not self.session:
                await self.initialize()
                
            # Check local cache first
            cache_file = f"data/raw/{symbol}_news.json"
            if os.path.exists(cache_file):
                file_age = time.time() - os.path.getmtime(cache_file)
                if file_age < 21600:  # Less than 6 hours old
                    try:
                        data = load_from_json(cache_file)
                        if data and 'feed' in data:
                            logger.info(f"Loading {symbol} news from local cache")
                            return self._process_news_data(data)
                    except Exception as e:
                        logger.error(f"Error loading cached news for {symbol}: {e}")
            
            # Apply rate limiting
            await self._rate_limit()
                
            url = (f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT"
                  f"&tickers={symbol}&apikey={self.alpha_vantage_key}")
            
            async with self.session.get(url) as response:
                if response.status != 200:
                    logger.warning(f"Error getting news for {symbol}: Status {response.status}")
                    return self._get_mock_news(symbol)
                
                data = await response.json()
                save_to_json(data, cache_file)
                
                if 'Information' in data:
                    logger.warning(f"AlphaVantage API limit hit when getting news: {data['Information']}")
                    # Try to use cached data
                    if os.path.exists(cache_file):
                        try:
                            data = load_from_json(cache_file)
                            if data and 'feed' in data:
                                logger.info(f"Using older cached news for {symbol}")
                                return self._process_news_data(data)
                        except Exception as e:
                            logger.error(f"Error loading cached news for {symbol}: {e}")
                    return self._get_mock_news(symbol)
                
                if 'feed' not in data:
                    logger.warning(f"No news data for {symbol}")
                    return self._get_mock_news(symbol)
                    
                return self._process_news_data(data)
                
        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {e}")
            return self._get_mock_news(symbol)
    
    def _process_news_data(self, data):
        """Process the AlphaVantage news data"""
        news_items = []
        for item in data['feed'][:10]:  # Limit to 10 news items
            news_items.append({
                'title': item.get('title', ''),
                'summary': item.get('summary', ''),
                'url': item.get('url', ''),
                'time_published': item.get('time_published', ''),
                'authors': item.get('authors', []),
                'sentiment_score': item.get('overall_sentiment_score', 0),
                'sentiment_label': item.get('overall_sentiment_label', 'Neutral')
            })
        
        return news_items
    
    def _get_mock_news(self, symbol):
        """Generate mock news when API fails"""
        headlines = [
            f"{symbol} Announces Strong Quarterly Results",
            f"Analysts Upgrade {symbol} Stock Rating",
            f"New Product Launch Could Boost {symbol} Revenue",
            f"Market Conditions May Impact {symbol}'s Growth",
            f"{symbol} CEO Discusses Future Strategy"
        ]
        
        news_items = []
        for i, headline in enumerate(headlines):
            sentiment = np.random.uniform(-0.3, 0.7)  # Slightly positive bias
            news_items.append({
                'title': headline,
                'summary': f"This is a mock summary for {headline} with generated content for testing purposes.",
                'url': f"https://example.com/news/{symbol.lower()}/{i}",
                'time_published': datetime.now().strftime("%Y%m%dT%H%M%S"),
                'authors': ["Mock Data Generator"],
                'sentiment_score': sentiment,
                'sentiment_label': "Positive" if sentiment > 0.2 else "Neutral" if sentiment > -0.2 else "Negative"
            })
        
        return news_items