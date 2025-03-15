from typing import Dict, List
import aiohttp
from src.utils import logger


class SentimentAnalyzer:    
    def __init__(self, llm_endpoint: str = "http://localhost:8080/generate"):
        self.llm_endpoint = llm_endpoint
        self.session = None
        logger.info("Sentiment analyzer initialized")
        
    async def initialize(self):
        self.session = aiohttp.ClientSession()
    
    async def close(self):
        if self.session:
            await self.session.close()
            
    async def analyze_text(self, text: str) -> float:
        if not self.session:
            await self.initialize()
        default_score = 0.0
        
        try:
            prompt = f"""
            Analyze the sentiment of the following news about stocks or financial markets.
            Return a single float value between -1.0 (extremely negative) and 1.0 (extremely positive).
            Just return the number without any other text or explanation.
            
            NEWS: {text}
            """
            
            payload = {
                "prompt": prompt,
                "temp": 0.1,  
                "max": 10     
            }
            
            async with self.session.post(self.llm_endpoint, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    sentiment_text = result.get("response", "0.0").strip()
                    try:
                        sentiment_score = float(sentiment_text)
                        return max(min(sentiment_score, 1.0), -1.0)
                    except (ValueError, TypeError):
                        logger.warning(f"Could not parse sentiment score from: {sentiment_text}")
                        return default_score
                else:
                    logger.warning(f"LLM endpoint returned status {response.status}")
                    return default_score
                
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            positive_words = ["up", "rise", "gain", "growth", "positive", "profit", "bullish", "surge"]
            negative_words = ["down", "fall", "loss", "decline", "negative", "debt", "bearish", "plunge"]
            
            text_lower = text.lower()
            positive_count = sum(text_lower.count(word) for word in positive_words)
            negative_count = sum(text_lower.count(word) for word in negative_words)
            
            if positive_count == negative_count:
                return 0.0
            else:
                total = positive_count + negative_count
                return (positive_count - negative_count) / total
                
    async def analyze_news_batch(self, news_items: List[Dict]) -> List[Dict]:
        results = []
        for item in news_items:
            title = item.get('title', '')
            description = item.get('description', '') or item.get('content', '')
            text = f"{title}. {description}"
            symbol = None
            if 'meta' in item and 'symbols' in item['meta']:
                symbols = item['meta']['symbols']
                if symbols and len(symbols) > 0:
                    symbol = symbols[0]['symbol']
            
            sentiment_score = await self.analyze_text(text)
            
            results.append({
                'title': title,
                'snippet': description[:150] + '...' if len(description) > 150 else description,
                'sentiment': sentiment_score,
                'symbol': symbol
            })
            
        return results
