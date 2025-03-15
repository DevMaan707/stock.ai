"""Test suite for the stock prediction system"""

import os
import sys

# Add src directory to Python path for tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from .test_helpers import *

# Test environment setup
TEST_ENV = os.getenv('TEST_ENV', 'testing')