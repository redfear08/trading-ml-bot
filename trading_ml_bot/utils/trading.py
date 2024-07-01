import pandas as pd
from utils.features import add_features

def suggest_minimum_balance(df, position_size):
    latest_price = df['close'].iloc[-1]
    min_balance = latest_price * position_size
    return min_balance

def backtest(df, model, initial_balance=100000, stop_loss=0.1, position_size=1, take_profit=0.2):
    balance = initial_balance
    in_position = False
    buy_price = 0
    trades = []

    df.reset_index(drop=True, inplace=True)
    df['Predicted_Signal'] = model.predict(df[['SMA_50', 'SMA_200', 'RSI', 'Bollinger_Upper', 'Bollinger_Lower', 'MACD', 'MACD_Signal', 'Stochastic_%K', 'Stochastic_%D', 'Momentum', 'VWAP']])

    for i in range(len(df)):
        if df['Predicted_Signal'][i] == 1 and not in_position:
            buy_price = df['close'][i]
            min_balance_required = suggest_minimum_balance(df, position_size)
            if balance >= min_balance_required:
                in_position = True
                balance -= buy_price * position_size
                print(f"Buy at {buy_price}")
                trades.append((df['date'][i], "BUY", buy_price))
            else:
                print("Insufficient balance to make the trade.")
        elif df['Predicted_Signal'][i] == 0 and in_position:
            sell_price = df['close'][i]
            balance += sell_price * position_size
            in_position = False
            print(f"Sell at {sell_price}")
            trades.append((df['date'][i], "SELL", sell_price))
        # Check for stop loss
        if in_position and df['close'][i] <= buy_price * (1 - stop_loss):
            sell_price = df['close'][i]
            balance += sell_price * position_size
            in_position = False
            print(f"Stop loss triggered. Sell at {sell_price}")
            trades.append((df['date'][i], "SELL (Stop Loss)", sell_price))
        # Check for take profit
        if in_position and df['close'][i] >= buy_price * (1 + take_profit):
            sell_price = df['close'][i]
            balance += sell_price * position_size
            in_position = False
            print(f"Take profit triggered. Sell at {sell_price}")
            trades.append((df['date'][i], "SELL (Take Profit)", sell_price))

    return balance, trades
