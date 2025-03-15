import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import asyncio
import colorama

from src.utils import logger
colorama.init()
load_dotenv()

# async def main():
#     parser = argparse.ArgumentParser(description='Stock Market Prediction System')
#     parser.add_argument('--symbols', type=str, nargs='+', default=['AAPL', 'GOOGL', 'MSFT', 'AMZN'],
#                       help='List of stock symbols to monitor')
#     parser.add_argument('--interval', type=int, default=60,
#                       help='Prediction interval in minutes')
#     args = parser.parse_args()
#     system = StockPredictionSystem(args.symbols)
#     while True:
#         try:
#             await system.run_prediction_cycle()
#             await system.validate_predictions()
#             if datetime.now().hour % 6 == 0 and datetime.now().minute < args.interval:
#                 await system.train_models()
#             logger.info(f"Waiting {args.interval} minutes until next prediction cycle...")
#             await asyncio.sleep(args.interval * 60)
            
#         except KeyboardInterrupt:
#             logger.info("Shutting down...")
#             break
#         except Exception as e:
#             logger.error(f"Error in main loop: {e}")
#             await asyncio.sleep(60)  # Wait a minute before retrying

# if __name__ == "__main__":
#     asyncio.run(main())


#!/usr/bin/env python3

import asyncio
import argparse
from core.stock_prediction_system import StockPredictionSystem

def parse_arguments():
    parser = argparse.ArgumentParser(description='Stock Market Prediction System')
    parser.add_argument('--symbols', type=str, nargs='+',
                      default=['AAPL', 'GOOGL', 'MSFT', 'AMZN'],
                      help='List of stock symbols to monitor')
    parser.add_argument('--interval', type=int, default=60,
                      help='Prediction interval in minutes')
    return parser.parse_args()
async def run(self):
    while True:
        try:
            await self.run_prediction_cycle()
            await self.validate_predictions()
            if datetime.now().hour % 6 == 0 and datetime.now().minute < 60:
                await self.train_models()
            logger.info(f"Waiting 60 minutes until next prediction cycle...")
            await asyncio.sleep(60 * 60)
            
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            await asyncio.sleep(60)

async def main():
    args = parse_arguments()
    system = StockPredictionSystem(args.symbols)
    await system.run()

if __name__ == "__main__":
    asyncio.run(main())