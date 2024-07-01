import pandas as pd
import joblib
from utils.features import add_features
from utils.trading import backtest, suggest_minimum_balance

# Load historical data
df = pd.read_csv('../data/historical_data.csv')
df = add_features(df)

# Load the trained model
model_path = '../models/best_model.pkl'
best_model = joblib.load(model_path)

# User inputs for trading
initial_balance = float(input("Enter initial balance: "))
stop_loss = float(input("Enter stop loss percentage (e.g., 0.1 for 10%): "))
take_profit = float(input("Enter take profit percentage (e.g., 0.2 for 20%): "))
position_size = float(input("Enter position size (number of units): "))

print("Backtesting the machine learning strategy.")

final_balance, trades = backtest(df, best_model, initial_balance=initial_balance, stop_loss=stop_loss, position_size=position_size, take_profit=take_profit)
print(f"Final balance after backtesting: {final_balance}")
