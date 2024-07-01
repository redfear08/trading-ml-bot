import pandas as pd
import numpy as np

def add_features(df):
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['SMA_200'] = df['close'].rolling(window=200).mean()
    df['RSI'] = calculate_rsi(df, 14)['RSI']
    df['Bollinger_Upper'] = calculate_bollinger_bands(df, 20)['Upper_Band']
    df['Bollinger_Lower'] = calculate_bollinger_bands(df, 20)['Lower_Band']
    df['MACD'] = calculate_macd(df)['MACD']
    df['MACD_Signal'] = calculate_macd(df)['Signal_Line']
    df['Stochastic_%K'] = calculate_stochastic_oscillator(df, 14)['%K']
    df['Stochastic_%D'] = calculate_stochastic_oscillator(df, 14)['%D']
    df['Momentum'] = df['close'] / df['close'].shift(10) - 1
    df['VWAP'] = calculate_vwap(df)['VWAP']
    df['Target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
    df.dropna(inplace=True)
    return df

def calculate_rsi(df, window=14):
    delta = df['close'].diff(1)
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def calculate_bollinger_bands(df, window=20):
    df['SMA'] = df['close'].rolling(window=window).mean()
    df['STD'] = df['close'].rolling(window=window).std()
    df['Upper_Band'] = df['SMA'] + (df['STD'] * 2)
    df['Lower_Band'] = df['SMA'] - (df['STD'] * 2)
    return df

def calculate_macd(df, short_window=12, long_window=26, signal_window=9):
    df['EMA_12'] = df['close'].ewm(span=short_window, adjust=False).mean()
    df['EMA_26'] = df['close'].ewm(span=long_window, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=signal_window, adjust=False).mean()
    return df

def calculate_stochastic_oscillator(df, window=14):
    df['L14'] = df['low'].rolling(window=window).min()
    df['H14'] = df['high'].rolling(window=window).max()
    df['%K'] = (df['close'] - df['L14']) * 100 / (df['H14'] - df['L14'])
    df['%D'] = df['%K'].rolling(window=3).mean()
    return df

def calculate_vwap(df):
    df['Cumulative_TP_Volume'] = (df['close'] * df['volume']).cumsum()
    df['Cumulative_Volume'] = df['volume'].cumsum()
    df['VWAP'] = df['Cumulative_TP_Volume'] / df['Cumulative_Volume']
    return df
