from colorama import Fore, Style
from tqdm import tqdm

class ConsoleUI:
 
    
    @staticmethod
    def print_header():
        print(f"\n{Fore.CYAN}{'=' * 80}")
        print(f"{Fore.CYAN}{'Advanced Stock Market Prediction System':^80}")
        print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}\n")
    
    @staticmethod
    def print_prediction(symbol: str, action: int, confidence: float, 
                        current_price: float, predicted_price: float):
        action_map = {0: f"{Fore.GREEN}BUY", 1: f"{Fore.RED}SELL", 2: f"{Fore.YELLOW}HOLD"}
        direction = "↑" if predicted_price > current_price else "↓"
        change_pct = abs(predicted_price - current_price) / current_price * 100
        
        print(f"{Fore.BLUE}{symbol:<6}{Style.RESET_ALL} | " 
              f"Current: ${current_price:.2f} | "
              f"Prediction: ${predicted_price:.2f} {direction} ({change_pct:.2f}%) | "
              f"Action: {action_map[action]}{Style.RESET_ALL} | "
              f"Confidence: {confidence:.2f}%")
    
    @staticmethod
    def print_model_stats(version: int, accuracy: float, mse: float, mape: float):
        print(f"\n{Fore.MAGENTA}Model Statistics (v{version}){Style.RESET_ALL}")
        print(f"Accuracy: {accuracy:.2f}% | MSE: {mse:.4f} | MAPE: {mape:.4f}")
    
    @staticmethod
    def print_validation_result(symbol: str, predicted_direction: str, 
                              actual_direction: str, reward: float):
        result = f"{Fore.GREEN}CORRECT" if predicted_direction == actual_direction else f"{Fore.RED}WRONG"
        print(f"{Fore.BLUE}{symbol:<6}{Style.RESET_ALL} | "
              f"Predicted: {predicted_direction} | "
              f"Actual: {actual_direction} | "
              f"Result: {result}{Style.RESET_ALL} | "
              f"Reward: {reward:.4f}")
    
    @staticmethod
    def progress_bar(description: str, iterable):
        return tqdm(iterable, desc=description, unit="item")
