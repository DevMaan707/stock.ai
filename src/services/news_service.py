import asyncio
import os
import time
from typing import Dict, List

import aiohttp
from src.utils.logger import logger
from src.utils.helpers import ensure_directory, save_to_json, load_from_json


class NewsProvider:
    def __init__(self):
        self.api_key = os.getenv('ALPHA_VANTAGE_API_KEY', 'MS0Q3VIDCHALT4MT')
        self.session = None
        self.last_api_call = 0
        self.min_call_interval = 15  # seconds between API calls
        ensure_directory('data/raw')
        logger.info("News provider initialized")
        
    async def initialize(self):
        self.session = aiohttp.ClientSession()
        
    async def close(self):
        """Close session when done"""
        if self.session:
            await self.session.close()
            
    async def _rate_limit(self):
        """Implement rate limiting for API calls"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call
        
        if time_since_last_call < self.min_call_interval:
            wait_time = self.min_call_interval - time_since_last_call
            logger.info(f"Rate limiting in news provider: waiting {wait_time:.2f} seconds")
            await asyncio.sleep(wait_time)
            
        self.last_api_call = time.time()
    
    async def get_stock_news(self, symbols: List[str], days: int = 3) -> List[Dict]:
        if not self.session:
            await self.initialize()
            
        # Check local cache first
        cache_file = f"data/raw/market_news.json"
        if os.path.exists(cache_file):
            file_age = time.time() - os.path.getmtime(cache_file)
            if file_age < 14400:  # Less than 4 hours old
                try:
                    data = load_from_json(cache_file)
                    if data and 'feed' in data:
                        logger.info("Loading market news from local cache")
                        return self._convert_to_standard_format(data)
                except Exception as e:
                    logger.error(f"Error loading cached market news: {e}")
            
        news_items = []
        try:
            # Apply rate limiting
            await self._rate_limit()
                
            # Join symbols with commas for AlphaVantage
            symbol_query = ','.join(symbols[:3])  # Limit to first 3 to avoid query length issues
            
            url = (f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT"
                  f"&tickers={symbol_query}&apikey={self.api_key}")
                  
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    save_to_json(data, cache_file)
                    
                    if 'Information' in data:
                        logger.warning(f"AlphaVantage API limit hit: {data['Information']}")
                        return self._get_mock_news(symbols)
                    
                    if 'feed' in data:
                        return self._convert_to_standard_format(data)
                    else:
                        logger.warning(f"No news feed found in AlphaVantage response")
                        return self._get_mock_news(symbols)
                else:
                    logger.error(f"AlphaVantage News API error: {response.status}")
                    return self._get_mock_news(symbols)
                    
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return self._get_mock_news(symbols)
    
    def _convert_to_standard_format(self, data):
        """Convert AlphaVantage news to standard format"""
        news_items = []
        
        if 'feed' not in data:
            return news_items
            
        for item in data['feed']:
            # Convert AlphaVantage format to our standard format
            news_item = {
                'title': item.get('title', ''),
                'description': item.get('summary', ''),
                'url': item.get('url', ''),
                'publishedAt': item.get('time_published', ''),
                'source': {'name': ', '.join(item.get('authors', []))},
                'sentiment': {
                    'score': item.get('overall_sentiment_score', 0),
                    'label': item.get('overall_sentiment_label', 'Neutral')
                },
                'ticker_sentiments': {}
            }
            
            # Add ticker-specific sentiments
            for ticker_sentiment in item.get('ticker_sentiment', []):
                ticker = ticker_sentiment.get('ticker', '')
                news_item['ticker_sentiments'][ticker] = {
                    'score': ticker_sentiment.get('ticker_sentiment_score', 0),
                    'label': ticker_sentiment.get('ticker_sentiment_label', 'Neutral')
                }
            
            news_items.append(news_item)
            
        return news_items
        
    def _get_mock_news(self, symbols):
        """Generate mock news when API fails"""
        news_items = []
        
        # General market headlines
        market_headlines = [
            "Market Shows Signs of Recovery After Recent Volatility",
            "Investors Eye Fed Decision on Interest Rates",
            "Tech Stocks Lead Market Rally",
            "Economic Data Points to Continued Growth",
            "Global Markets Respond to Geopolitical Tensions"
        ]
        
        # Add general market news
        for i, headline in enumerate(market_headlines):
            sentiment_score = 0.1 * (i - 2)  # Range from -0.2 to 0.2
            
            news_item = {
                'title': headline,
                'description': f"This is mock market news content for testing purposes.",
                'url': f"https://example.com/market-news/{i}",
                'publishedAt': time.strftime("%Y%m%dT%H%M%S", time.localtime(time.time() - i*3600)),
                'source': {'name': "Mock Financial News"},
                'sentiment': {
                    'score': sentiment_score,
                    'label': "Positive" if sentiment_score > 0.1 else "Neutral" if sentiment_score > -0.1 else "Negative"
                },
                'ticker_sentiments': {}
            }
            
            news_items.append(news_item)
        
        # Add stock-specific news
        for i, symbol in enumerate(symbols[:3]):  # Just handle first 3 symbols
            headline = f"{symbol} Announces New Strategic Initiative"
            sentiment_score = 0.3  # Positive bias for company news
            
            ticker_sentiments = {}
            ticker_sentiments[symbol] = {
                'score': sentiment_score,
                'label': "Positive"
            }
            
            news_item = {
                'title': headline,
                'description': f"{symbol} today announced a new strategic initiative aimed at increasing shareholder value.",
                'url': f"https://example.com/stock-news/{symbol.lower()}",
                'publishedAt': time.strftime("%Y%m%dT%H%M%S", time.localtime(time.time() - i*7200)),
                'source': {'name': "Mock Company News"},
                'sentiment': {
                    'score': sentiment_score,
                    'label': "Positive"
                },
                'ticker_sentiments': ticker_sentiments
            }
            
            news_items.append(news_item)
        
        return news_items
        
    async def get_news_by_ticker(self, symbol: str) -> List[Dict]:
        """Get news specifically for one ticker"""
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
                        return self._extract_ticker_news(data, symbol)
                except Exception as e:
                    logger.error(f"Error loading cached news for {symbol}: {e}")
        
        try:
            # Apply rate limiting
            await self._rate_limit()
                
            url = (f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT"
                  f"&tickers={symbol}&apikey={self.api_key}")
                  
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    save_to_json(data, cache_file)
                    
                    if 'Information' in data:
                        logger.warning(f"AlphaVantage API limit hit for {symbol} news: {data['Information']}")
                        return self._get_mock_ticker_news(symbol)
                    
                    if 'feed' in data:
                        return self._extract_ticker_news(data, symbol)
                    else:
                        logger.warning(f"No news feed found for {symbol}")
                        return self._get_mock_ticker_news(symbol)
                else:
                    logger.error(f"AlphaVantage News API error for {symbol}: {response.status}")
                    return self._get_mock_ticker_news(symbol)
                    
        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {e}")
            return self._get_mock_ticker_news(symbol)
    
    def _extract_ticker_news(self, data, symbol):
        """Extract news items for a specific ticker"""
        news_items = []
        
        if 'feed' not in data:
            return news_items
            
        for item in data['feed']:
            # Find ticker-specific sentiment if available
            sentiment_score = item.get('overall_sentiment_score', 0)
            for ticker_sentiment in item.get('ticker_sentiment', []):
                if ticker_sentiment.get('ticker') == symbol:
                    sentiment_score = ticker_sentiment.get('ticker_sentiment_score', sentiment_score)
                    break
            
            news_items.append({
                'title': item.get('title', ''),
                'description': item.get('summary', ''),
                'url': item.get('url', ''),
                'publishedAt': item.get('time_published', ''),
                'source': {'name': ', '.join(item.get('authors', []))},
                'sentiment_score': sentiment_score
            })
        
        return news_items
    
    def _get_mock_ticker_news(self, symbol):
        """Generate mock ticker-specific news"""
        headlines = [
            f"{symbol} Reports Quarterly Earnings Above Expectations",
            f"Analyst Upgrades {symbol} Rating to 'Buy'",
            f"{symbol} Expands into New Markets",
            f"{symbol} Announces Leadership Changes",
            f"Investors React to {symbol}'s Latest Product Release"
        ]
        
        news_items = []
        for i, headline in enumerate(headlines):
            sentiment = 0.2 + (i * 0.1)  # Positive sentiment for company news
            if i >= 3:  # Make some news items negative
                sentiment = -0.2 - ((i-3) * 0.1)
                
            # Create date strings in a format compatible with our system
            date_str = time.strftime("%Y-%m-%d", time.localtime(time.time() - i*86400))
            
            news_items.append({
                'title': headline,
                'description': f"This is mock content for {symbol} news testing.",
                'url': f"https://example.com/{symbol.lower()}/news/{i}",
                'publishedAt': time.strftime("%Y%m%dT%H%M%S", time.localtime(time.time() - i*86400)),
                'source': {'name': "Mock Financial News"},
                'sentiment_score': sentiment
            })
        
        return news_items