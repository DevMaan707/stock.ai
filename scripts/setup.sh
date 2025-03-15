#!/bin/bash

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Create necessary directories
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/database
mkdir -p models/trained
mkdir -p models/checkpoints
mkdir -p logs

# Initialize database
export PYTHONPATH=.
python src/core/database.py

echo "Setup completed successfully!"