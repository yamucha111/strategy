import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from env.BacktestEnv import BacktestEnv

from utils.Utils import Utils
from core import data_fetcher

if __name__ == "__main__":
    symbol = 'BTCUSDT'
    interval_fast = '1h'
    interval_slow = '1d'
    
    start_date = '2019-03-01'
    end_date = '2024-12-31'

    fast_his_data = data_fetcher.query_klines(symbol, interval_fast, start_date, end_date)
    slow_his_data = data_fetcher.query_klines(symbol, interval_slow, start_date, end_date)
    
    fast_his_data = Utils.str_to_numeric(fast_his_data)
    slow_his_data = Utils.str_to_numeric(slow_his_data)
    
    fast_his_data['date'] = pd.to_datetime(fast_his_data['timestamp'])
    slow_his_data['date'] = pd.to_datetime(slow_his_data['timestamp'])

    # 实例化回测环境
    env = BacktestEnv(fast_his_data, slow_his_data, fast_interval=interval_fast, slow_interval=interval_slow, speed=100, window=160)

    # 添加交易信号
    # env.add_buy_signal(pd.Timestamp('2021-01-05 12:34'), 35000)
    # env.add_sell_signal(pd.Timestamp('2021-01-07 15:20'), 37000)

    # 运行回测环境
    env.run()
