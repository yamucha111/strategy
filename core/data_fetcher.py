# data_fetcher.py

import requests, os
import pandas as pd
import time, datetime

from utils.Utils import Utils
from config import BINANCE_API_URL, LOCAL_DATA_DIR

def get_historical_klines(symbol, interval, start_str, end_str=None, limit=500):
    url = f"{BINANCE_API_URL}/api/v3/klines"
    all_data = []
    start_time = start_str
    retry_attempts = 3

    while True:
        try:
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': start_time,
                'limit': limit
            }
            if end_str:
                params['endTime'] = end_str

            response = requests.get(url, params=params)
            data = response.json()
            
            if not data:
                print(f"No data fetched: {data}")
                break
                        
            start_timestamp = data[0][0]
            end_timestamp = data[-1][0]
            start_dt = Utils.timestamp_to_date_string(start_timestamp / 1000)
            end_dt = Utils.timestamp_to_date_string(end_timestamp / 1000)
            print(f"当前获取到数据：{start_dt} - {end_dt}...")
            
            all_data.extend(data)
            
            # Update start_time for next batch
            start_time = data[-1][0] + 1

            # Check if we have reached the end_time
            if end_str and data[-1][0] >= end_str:
                break
            
            # Sleep to avoid hitting API rate limits
            time.sleep(Utils.generate_random_number(1) + 1)
            retry_attempts = 3
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            if retry_attempts > 0:
                print(f"Error fetching data: {e}. Retrying... ({retry_attempts} attempts left)")
                retry_attempts -= 1
                time.sleep(Utils.generate_random_number(1))
                continue
            else:
                print(f"Error fetching data: {e}. Maximum retry attempts reached.")
                break
        except Exception as e:
            print(f"Unexpected error fetching data: {e}")
            break
        
    _columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']

    if all_data:
        df = pd.DataFrame(all_data, columns=_columns)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    else:
        return pd.DataFrame(columns=_columns)

def query_klines(symbol, interval, start_date, end_date=None, sort=True):
    current_time = datetime.datetime.now()
    start_timestamp = int(pd.Timestamp(start_date).timestamp() * 1000)
    end_timestamp = int(pd.Timestamp(end_date).timestamp() * 1000) if end_date else int(pd.Timestamp(datetime.now()).timestamp() * 1000)
    
    if end_date and pd.Timestamp(end_date) > current_time:
        end_date = current_time
        end_timestamp = int(current_time.timestamp() * 1000)

    current_year = Utils.timestamp_to_date(start_timestamp / 1000).year
    end_year = Utils.timestamp_to_date(end_timestamp / 1000).year

    all_data = []

    for year in range(current_year, end_year + 1):
        filename = f"{symbol}_{interval}_{year}.csv"
        df = load_from_csv(filename)
        if df is not None:
            all_data.append(df)
    
    if all_data:
        combined_df = pd.concat(all_data)
        combined_df = combined_df[(combined_df['timestamp'] >= pd.to_datetime(start_date)) & (combined_df['timestamp'] <= pd.to_datetime(end_date))]
    else:
        combined_df = pd.DataFrame()

    if combined_df.empty or combined_df['timestamp'].min() > pd.to_datetime(start_date) or combined_df['timestamp'].max() < pd.to_datetime(end_date):
        missing_data = get_historical_klines(symbol, interval, start_timestamp, end_timestamp)
        if not missing_data.empty:
            combined_df = pd.concat([combined_df, missing_data]).drop_duplicates().reset_index(drop=True)
            for year in range(current_year, end_year + 1):
                year_data = combined_df[(combined_df['timestamp'].dt.year == year)]
                if not year_data.empty:
                    save_to_csv(year_data, f"{symbol}_{interval}_{year}.csv")

    if sort:
        combined_df = combined_df.sort_values(by='timestamp')

    return combined_df

def save_to_csv(df, filename):
    if not os.path.exists(LOCAL_DATA_DIR):
        os.makedirs(LOCAL_DATA_DIR)
    filepath = os.path.join(LOCAL_DATA_DIR, filename)
    df.to_csv(filepath, index=False)

def load_from_csv(filename):
    filepath = os.path.join(LOCAL_DATA_DIR, filename)
    if os.path.exists(filepath):
        return pd.read_csv(filepath, parse_dates=['timestamp'])
    return None
