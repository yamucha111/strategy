
from utils.Utils import Utils
from core import data_fetcher

if __name__ == "__main__":
    symbol = 'BTCUSDT'
    interval = '1m'
    start_date = '2021-01-01'
    end_date = '2021-12-31'

    df = data_fetcher.query_klines(symbol, interval, start_date, end_date)

    print(df)
