# Stock.ai - Advanced Stock Market Prediction System

![Stock.ai Logo](https://img.shields.io/badge/Stock.ai-1.0.0-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

Stock.ai is a comprehensive platform for predicting stock market movements using machine learning models, real-time data analysis, and sentiment analysis of financial news. The system continuously learns from its predictions to improve accuracy over time.

## Features

- **Real-time Stock Data Analysis**: Fetch and process stock data from Yahoo Finance
- **Advanced ML Predictions**: LSTM neural networks for time series forecasting
- **Technical Indicators**: RSI, MACD, Bollinger Bands, and more
- **Sentiment Analysis**: Analyze financial news for market sentiment
- **Recommendation Engine**: Generate actionable buy/sell recommendations
- **Performance Tracking**: Track and validate prediction accuracy
- **Continuous Learning**: Models that improve with each prediction

## System Architecture

```
├── core/               # Core system components
│   ├── data_manager.py      # Fetches and manages stock data
│   ├── database.py          # Handles data storage and retrieval
│   ├── feature_engineer.py  # Creates features for ML models
│   ├── model_manager.py     # Manages ML model lifecycle
│   ├── recommendation_system.py  # Generates actionable recommendations
│   └── stock_prediction_system.py  # Main system orchestration
├── models/             # ML model definitions
├── services/           # External service integrations
│   ├── news_service.py      # Fetches financial news
│   ├── sentiment_service.py # Analyzes news sentiment
│   └── llm_service.py       # Large language model integration
├── ui/                 # User interface components
└── utils/              # Utility functions and helpers
```

## How It Works

### Workflow

1. **Data Collection**:
   - Stock price data is fetched from Yahoo Finance
   - Financial news is gathered from News API
   - Data is processed and stored in the local database

2. **Feature Engineering**:
   - Technical indicators are calculated (RSI, MACD, etc.)
   - News sentiment is analyzed and integrated with price data
   - Features are normalized for model input

3. **Prediction Generation**:
   - LSTM models process feature data to predict future prices
   - Confidence scores are calculated for each prediction
   - Buy/sell/hold recommendations are determined

4. **Result Validation**:
   - Previous predictions are compared with actual outcomes
   - Model performance metrics are updated
   - Rewards/penalties are assigned to improve future predictions

5. **Continuous Learning**:
   - Models are retrained periodically (default: every 6 hours)
   - New market data and prediction results enhance model accuracy
   - System performance improves over time

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- TA-Lib installed on your system (for technical indicators)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/stock.ai.git
   cd stock.ai
   ```

2. Run the setup script:
   ```bash
   bash scripts/setup.sh
   ```
   
   This will:
   - Create a virtual environment
   - Install required packages
   - Set up the database and directories

3. Configure your `.env` file:
   ```
   NEWS_API_KEY=your_news_api_key
   LLM_ENDPOINT=http://localhost:8080/generate
   DATABASE_PATH=data/stock_predictor.db
   MODEL_PATH=models/
   LOG_PATH=logs/
   ```

## Usage

### Basic Run

To start the system with default settings:

```bash
bash scripts/run.sh
```

Or run directly with Python:

```bash
python src/main.py --symbols AAPL GOOGL MSFT AMZN --interval 60
```

### Command Line Arguments

- `--symbols`: List of stock symbols to monitor (default: AAPL GOOGL MSFT AMZN)
- `--interval`: Prediction interval in minutes (default: 60)

### Changing Run Frequency

To adjust how often the system makes predictions:

```bash
python src/main.py --symbols AAPL GOOGL MSFT --interval 30
```

This will run prediction cycles every 30 minutes instead of the default 60.

### Training Schedule

By default, models are retrained every 6 hours. To change this, modify the `run` method in `src/core/stock_prediction_system.py`:

```python
if datetime.now().hour % 12 == 0:  # Change from 6 to 12 for 12-hour retraining
    await self.train_models()
```

## Monitoring

- Predictions and system status are displayed in the console
- Detailed logs are saved to `logs/stock_predictor.log`
- Database stores all predictions and performance metrics

## Data Storage

- **Raw Data**: Stored in `data/raw/`
- **Processed Data**: Stored in `data/processed/`
- **Database**: SQLite database at `data/stock_predictor.db`
- **Models**: Saved to `models/` directory

## Advanced Configuration

Edit `src/utils/config.py` to change system-wide settings:

- `LOOKBACK_DAYS`: Days of historical data to use (default: 500)
- `TRAINING_EPOCHS`: Number of epochs for model training (default: 10)
- `BATCH_SIZE`: Batch size for training (default: 32)
- `TECHNICAL_INDICATORS`: List of technical indicators to use

## Extending the System

### Adding New Stock Symbols

Simply add them to the command line arguments:

```bash
python src/main.py --symbols AAPL GOOGL MSFT AMZN TSLA NFLX
```

### Adding Custom Technical Indicators

1. Extend the `add_technical_indicators` method in `src/core/feature_engineer.py`
2. Update the `feature_list` in the `FeatureEngineer` class
3. Retrain models to incorporate the new features

### Customizing Recommendation Logic

Modify the `analyze_stock` method in `src/core/recommendation_system.py` to adjust how recommendations are generated.

## Troubleshooting

- **Missing Data**: Ensure your internet connection is stable for API access
- **Model Training Errors**: Check if you have enough historical data for selected symbols
- **TA-Lib Errors**: Verify TA-Lib is properly installed on your system

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Yahoo Finance for stock data
- NewsAPI for financial news
- TensorFlow and Keras for machine learning capabilities

---