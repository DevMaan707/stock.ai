#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Run the main application
python src/main.py --symbols AAPL GOOGL MSFT AMZN --interval 30