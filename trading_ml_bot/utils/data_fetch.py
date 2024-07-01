import os
from kiteconnect import KiteConnect
import datetime as dt
import pandas as pd

# API credentials
api_key = "klz728yv89qrljzs"
api_secret = "4vhxunujbp17i8da0y1tiy7ayde4h5o8"

# Authenticate and get access token
kite = KiteConnect(api_key=api_key)
print("Login URL:", kite.login_url())
request_token = input("Enter request token: ")
data = kite.generate_session(request_token, api_secret=api_secret)
access_token = data["access_token"]
kite.set_access_token(access_token)

def fetch_historical_data(instrument_token, from_date, to_date, interval):
    return kite.historical_data(instrument_token, from_date, to_date, interval)

# Fetch historical data for a specific stock (e.g., NSE: INFY)
instrument_token = 738561  # Example token for NSE: INFY
from_date = dt.date(2023, 11, 10)
to_date = dt.date(2023, 12, 31)
interval = "minute"

historical_data = fetch_historical_data(instrument_token, from_date, to_date, interval)
df = pd.DataFrame(historical_data)

# Create data directory if it does not exist
data_dir = '../data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Save data to CSV
df.to_csv(os.path.join(data_dir, 'historical_data.csv'), index=False)

print("Historical data fetched and saved successfully.")
