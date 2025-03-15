import logging.handlers
import os
os.makedirs('logs', exist_ok=True)
log_handler = logging.handlers.RotatingFileHandler(
    'logs/stock_predictor.log', maxBytes=10485760, backupCount=5)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[log_handler, console_handler]
)
logger = logging.getLogger(__name__)
