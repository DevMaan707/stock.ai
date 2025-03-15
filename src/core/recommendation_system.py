from typing import Dict
import aiohttp
from src.core.data_manager import DataManager
from src.core.database import Database
from src.core.model_manager import ModelManager
from src.utils import logger
import yfinance as yf

class RecommendationEngine: 
    def __init__(self, db: Database, model_manager: ModelManager, data_manager: DataManager):
        self.db = db
        self.model_manager = model_manager
        self.data_manager = data_manager
        self.llm_endpoint = "http://localhost:8080/generate"
        logger.info("Recommendation engine initialized")
        
    async def analyze_stock(self, symbol: str, prediction: Dict, 
                          sentiment_score: float = 0.0) -> Dict:
        try:
            accuracy = self.db.get_prediction_accuracy(symbol, "week")
            expected_return = prediction['price_change_pct']
            df = await self.data_manager.fetch_stock_data(symbol)
            if df.empty:
                risk_score = 50
            else:
                returns = df['Close'].pct_change().dropna()
                volatility = returns.std() * (252 ** 0.5) * 100 
                risk_score = min(volatility * 5, 100) 
            if expected_return > 2 and prediction['confidence'] > 70:
                recommendation = "Strong Buy"
            elif expected_return > 0.5:
                recommendation = "Buy"
            elif expected_return < -2 and prediction['confidence'] > 70:
                recommendation = "Strong Sell"
            elif expected_return < -0.5:
                recommendation = "Sell"
            else:
                recommendation = "Hold"
            reasons = await self._generate_reasons(symbol, prediction, sentiment_score, risk_score)
            
            return {
                'symbol': symbol,
                'expected_return': expected_return,
                'risk_score': risk_score,
                'confidence': prediction['confidence'],
                'model_accuracy': accuracy,
                'recommendation': recommendation,
                'reasons': reasons
            }
            
        except Exception as e:
            logger.error(f"Error analyzing stock {symbol}: {e}")
            return {
                'symbol': symbol,
                'recommendation': "Hold",
                'reasons': "Insufficient data for analysis"
            }
    
    async def _generate_reasons(self, symbol: str, prediction: Dict, 
                              sentiment_score: float, risk_score: float) -> str:
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            company_name = info.get('shortName', symbol)
            sector = info.get('sector', 'Unknown')
            
            prompt = f"""
            Generate a brief analysis (3-4 sentences) explaining a stock recommendation for {company_name} ({symbol}) in the {sector} sector.
            
            Key factors:
            - Current price: ${prediction['current_price']:.2f}
            - Predicted price: ${prediction['predicted_price']:.2f} ({prediction['price_change_pct']:.2f}% change)
            - Prediction confidence: {prediction['confidence']:.1f}%
            - News sentiment: {"Positive" if sentiment_score > 0.2 else "Negative" if sentiment_score < -0.2 else "Neutral"} ({sentiment_score:.2f})
            - Risk level: {"High" if risk_score > 70 else "Medium" if risk_score > 40 else "Low"} ({risk_score:.1f}/100)
            
            Based on these factors, the recommendation is: {prediction['action'] if prediction['action'] in [0, 1] else 'Hold'}
            
            Keep your analysis factual, logical, and data-driven. DO NOT include price targets or specific return percentages.
            """
            
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.post(self.llm_endpoint, json={
                        "prompt": prompt,
                        "temp": 0.3,
                        "max": 200
                    }) as response:
                        response_data = await response.json()
                        analysis = response_data.get('generated_text', '').strip()
                        return analysis
                except Exception as e:
                    logger.error(f"Error getting LLM analysis: {e}")
                    template = (
                        f"Analysis based on quantitative factors: "
                        f"{'Positive' if prediction['price_change_pct'] > 0 else 'Negative'} price momentum "
                        f"with {'high' if prediction['confidence'] > 70 else 'moderate'} confidence. "
                        f"{'Favorable' if sentiment_score > 0 else 'Unfavorable'} market sentiment "
                        f"and {'high' if risk_score > 70 else 'moderate' if risk_score > 40 else 'low'} risk profile."
                    )
                    return template
                    
        except Exception as e:
            logger.error(f"Error generating analysis for {symbol}: {e}")
            return "Unable to generate detailed analysis due to technical issues."
