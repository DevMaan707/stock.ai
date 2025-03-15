import aiohttp
import os
import json
from typing import Dict, List, Union, Any
from src.utils.logger import logger

class LLMService:
    """Service for interacting with Language Learning Models"""
    
    def __init__(self, endpoint: str = None):
        """
        Initialize the LLM service
        
        Args:
            endpoint: API endpoint URL (default: use environment variable)
        """
        self.endpoint = endpoint or os.getenv('LLM_ENDPOINT', 'http://localhost:8080/generate')
        self.session = None
        logger.info(f"LLM Service initialized with endpoint: {self.endpoint}")
        
    async def initialize(self):
        """Initialize the HTTP session"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
            
    async def close(self):
        """Close the HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
            
    async def generate(self, prompt: str, max_tokens: int = 100, 
                    temperature: float = 0.7) -> str:
        """
        Generate text using the LLM
        
        Args:
            prompt: The input prompt for the model
            max_tokens: Maximum number of tokens to generate
            temperature: Model temperature (randomness)
            
        Returns:
            Generated text string
        """
        if not self.session:
            await self.initialize()
            
        try:
            payload = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            async with self.session.post(self.endpoint, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    # Handle different API response formats
                    if "generated_text" in result:
                        return result["generated_text"]
                    elif "response" in result:
                        return result["response"]
                    elif "text" in result:
                        return result["text"]
                    else:
                        return str(result)
                else:
                    error_text = await response.text()
                    logger.error(f"LLM API error: {response.status} - {error_text}")
                    return f"Error: {response.status}"
                    
        except Exception as e:
            logger.error(f"LLM service error: {e}")
            return f"Error: {str(e)}"
            
    async def analyze_sentiment(self, text: str) -> float:
        """
        Analyze sentiment using the LLM
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment score between -1.0 and 1.0
        """
        prompt = f"""
        Analyze the sentiment of the following text related to stocks or financial markets.
        Return only a single number between -1.0 (extremely negative) and 1.0 (extremely positive).
        
        Text: {text}
        
        Sentiment score:
        """
        
        try:
            response = await self.generate(prompt, max_tokens=10, temperature=0.1)
            # Try to extract a float from the response
            for word in response.split():
                try:
                    value = float(word)
                    if -1.0 <= value <= 1.0:
                        return value
                except ValueError:
                    continue
                    
            # If no valid float was found
            logger.warning(f"Could not extract sentiment score from: {response}")
            return 0.0
            
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return 0.0
            
    async def summarize_news(self, text: str, max_length: int = 100) -> str:
        """
        Summarize news text
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary
            
        Returns:
            Summarized text
        """
        prompt = f"""
        Summarize the following financial news briefly:
        
        {text}
        
        Concise summary:
        """
        
        return await self.generate(prompt, max_tokens=max_length, temperature=0.5)
        
    async def generate_stock_analysis(self, symbol: str, 
                                   price_data: Dict,
                                   sentiment: float) -> str:
        """
        Generate analysis for a stock
        
        Args:
            symbol: Stock symbol
            price_data: Price and trend data
            sentiment: Sentiment score
            
        Returns:
            Analysis text
        """
        prompt = f"""
        Generate a brief analysis (3-4 sentences) for {symbol} stock based on the following data:
        
        Current price: ${price_data.get('current_price', 'N/A')}
        Price change: {price_data.get('price_change_pct', 'N/A')}%
        Market sentiment: {sentiment}
        
        Analysis:
        """
        
        return await self.generate(prompt, max_tokens=150, temperature=0.7)