import numpy as np
from collections import defaultdict

class SentimentModel:
    """Model for sentiment analysis on financial texts"""
    
    def __init__(self):
        """Initialize the sentiment model"""
        # Positive and negative word lists for simple rule-based sentiment analysis
        self.positive_words = {
            'bullish', 'uptrend', 'growth', 'grow', 'increase', 'increasing',
            'profit', 'profitable', 'gains', 'gain', 'positive', 'up',
            'surged', 'surges', 'surge', 'rising', 'rise', 'rose', 'rises',
            'outperform', 'outperformed', 'beat', 'beats', 'exceeded',
            'exceed', 'exceeds', 'strong', 'stronger', 'strongest',
            'opportunity', 'opportunities', 'optimistic', 'optimism',
            'success', 'successful', 'succeed', 'succeeds', 'succeeded',
            'recovery', 'recover', 'recovers', 'recovered', 'winning', 'win'
        }
        
        self.negative_words = {
            'bearish', 'downtrend', 'decline', 'declining', 'decrease',
            'decreasing', 'loss', 'losses', 'negative', 'down', 'dropped',
            'drops', 'drop', 'falling', 'fall', 'fell', 'falls', 'underperform',
            'underperformed', 'miss', 'missed', 'misses', 'weak', 'weaker',
            'weakest', 'risk', 'risks', 'risky', 'pessimistic', 'pessimism',
            'fail', 'fails', 'failed', 'failure', 'downturn', 'crash',
            'bearish', 'recession', 'concern', 'concerned', 'concerns', 'warning'
        }
        
        # Word intensity modifiers
        self.intensifiers = {
            'very': 1.5, 'extremely': 2.0, 'significantly': 1.7, 
            'substantially': 1.7, 'notably': 1.3, 'considerably': 1.5,
            'highly': 1.5, 'majorly': 1.5, 'strongly': 1.5
        }
        
        # Stock-specific sentiment cache to avoid reprocessing
        self.sentiment_cache = {}
        
    def analyze(self, text, symbol=None):
        """
        Analyze sentiment of financial text
        
        Args:
            text: The text to analyze
            symbol: Optional stock symbol for context
            
        Returns:
            Float between -1.0 (negative) and 1.0 (positive)
        """
        # Check cache first
        cache_key = f"{symbol}:{text[:50]}" if symbol else text[:50]
        if cache_key in self.sentiment_cache:
            return self.sentiment_cache[cache_key]
        
        if not text:
            return 0.0
            
        # Convert to lowercase for matching
        text_lower = text.lower()
        words = text_lower.split()
        
        # Count positive and negative words
        pos_count = 0
        neg_count = 0
        
        # Track word positions for proximity analysis
        word_positions = defaultdict(list)
        
        for i, word in enumerate(words):
            word = word.strip('.,!?;:()"\'')
            if word in self.positive_words:
                pos_count += 1
                word_positions['positive'].append(i)
            elif word in self.negative_words:
                neg_count += 1
                word_positions['negative'].append(i)
        
        # Apply intensifiers if present
        for i, word in enumerate(words):
            if word in self.intensifiers:
                # Look ahead for sentiment words to intensify
                for j in range(i+1, min(i+4, len(words))):
                    intensified_word = words[j].strip('.,!?;:()"\'')
                    if intensified_word in self.positive_words:
                        pos_count += self.intensifiers[word] - 1
                        break
                    elif intensified_word in self.negative_words:
                        neg_count += self.intensifiers[word] - 1
                        break
        
        # Check for negations (simple approach)
        negations = ['not', 'no', "n't", 'never', 'neither', 'nor', 'none']
        for i, word in enumerate(words):
            if word in negations:
                # Look ahead for sentiment words to negate
                for j in range(i+1, min(i+4, len(words))):
                    negated_word = words[j].strip('.,!?;:()"\'')
                    if negated_word in self.positive_words:
                        pos_count -= 1
                        neg_count += 1
                        break
                    elif negated_word in self.negative_words:
                        neg_count -= 1
                        pos_count += 1
                        break
        
        # Calculate sentiment score
        total_count = pos_count + neg_count
        if total_count == 0:
            sentiment = 0.0
        else:
            sentiment = (pos_count - neg_count) / total_count
            
        # Cache the result
        self.sentiment_cache[cache_key] = sentiment
        
        return sentiment
    
    def batch_analyze(self, texts, symbols=None):
        """
        Analyze sentiment for multiple texts
        
        Args:
            texts: List of texts to analyze
            symbols: Optional list of corresponding symbols
            
        Returns:
            List of sentiment scores
        """
        results = []
        for i, text in enumerate(texts):
            symbol = symbols[i] if symbols and i < len(symbols) else None
            results.append(self.analyze(text, symbol))
        return results