"""
Jupyter notebooks for analysis and development
Note: This __init__.py is optional for notebooks directory
"""

import sys
import os

# Add src directory to Python path for notebooks
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))