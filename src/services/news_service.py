import os
from typing import Dict, List

import aiohttp
from src.utils import logger


class NewsProvider:
    def __init__(self):
        self.api_key = os.getenv('NEWS_API_KEY', 'demo') 
        self.session = None
        logger.info("News provider initialized")
        
    async def initialize(self):
        self.session = aiohttp.ClientSession()
        
    async def close(self):
        """Close session when done"""
        if self.session:
            await self.session.close()
    
    async def get_stock_news(self, symbols: List[str], days: int = 3) -> List[Dict]:
        if not self.session:
            await self.initialize()
            
        news_items = []
        try:
            symbol_query = ' OR '.join(symbols)
            async with self.session.get(
                f"https://newsapi.org/v2/everything?"
                f"q={symbol_query}&language=en&sortBy=publishedAt&pageSize=20"
                f"&apiKey={self.api_key}"
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('status') == 'ok':
                        news_items.extend(data.get('articles', []))
            if len(news_items) < 10:
                async with self.session.get(
                    f"https://newsapi.org/v2/everything?"
                    f"q=stock market OR finance&language=en&sortBy=publishedAt&pageSize=10"
                    f"&apiKey={self.api_key}"
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('status') == 'ok':
                            news_items.extend(data.get('articles', []))
                            
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            try:
                symbol_str = ','.join(symbols[:3])
                async with self.session.get(
                    f"https://api.marketaux.com/v1/news/all?"
                    f"symbols={symbol_str}&filter_entities=true&language=en&api_token=demo"
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        news_items.extend(data.get('data', []))
            except Exception as e2:
                logger.error(f"Error with fallback news API: {e2}")
        
        return news_items