# websocket_client.py

import websocket
import json
import threading
import time
import pandas as pd
from core.data_fetcher import get_historical_klines, save_to_csv, load_from_csv
from config import BINANCE_API_URL, KLINE_INTERVALS, LOCAL_DATA_DIR

class BinanceWebSocket:
    def __init__(self, symbol, interval):
        self.symbol = symbol
        self.interval = interval
        self.url = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@kline_{interval}"
        self.data = []

    def on_message(self, ws, message):
        json_message = json.loads(message)
        kline = json_message['k']
        is_kline_closed = kline['x']
        if is_kline_closed:
            kline_data = {
                'timestamp': pd.to_datetime(kline['t'], unit='ms'),
                'open': float(kline['o']),
                'high': float(kline['h']),
                'low': float(kline['l']),
                'close': float(kline['c']),
                'volume': float(kline['v'])
            }
            self.data.append(kline_data)

    def on_error(self, ws, error):
        print(f"Error: {error}")

    def on_close(self, ws):
        print("WebSocket closed")

    def on_open(self, ws):
        print("WebSocket opened")

    def start(self):
        ws = websocket.WebSocketApp(self.url, on_message=self.on_message, on_error=self.on_error, on_close=self.on_close)
        ws.on_open = self.on_open
        ws.run_forever()

def start_websocket(symbol, interval):
    ws = BinanceWebSocket(symbol, interval)
    ws_thread = threading.Thread(target=ws.start)
    ws_thread.start()
    return ws
