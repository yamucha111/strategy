import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime, timedelta

class BacktestEnv:
    def __init__(self, minute_data, higher_data, fast_interval='1m', slow_interval='1h', speed=100, window=50):
        self.minute_data = minute_data
        self.higher_data = higher_data
        self.fast_interval = fast_interval
        self.slow_interval = slow_interval
        self.speed = speed
        self.window = window
        self.interval_factor = self.get_interval_factor()
        self.current_index = self.window * self.interval_factor  # 从第window根高时间单位K线开始
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        self.buy_signals = []
        self.sell_signals = []
        self.accumulated_volume = 0  # 用于累积当前高时间单位K线的交易量
        self.previous_higher_time = None  # 用于记录上一个高时间单位时间戳

    def get_interval_minutes(self, interval):
        """将时间单位转化为分钟数"""
        if interval.endswith('m'):
            return int(interval[:-1])
        elif interval.endswith('h'):
            return int(interval[:-1]) * 60
        elif interval.endswith('d'):
            return int(interval[:-1]) * 60 * 24
        elif interval.endswith('w'):
            return int(interval[:-1]) * 60 * 24 * 7
        elif interval.endswith('M'):
            return int(interval[:-1]) * 60 * 24 * 30
        else:
            raise ValueError(f"Unsupported interval: {interval}")

    def get_interval_factor(self):
        """计算小时间单位在大时间单位内的循环次数"""
        fast_minutes = self.get_interval_minutes(self.fast_interval)
        slow_minutes = self.get_interval_minutes(self.slow_interval)
        if slow_minutes % fast_minutes != 0:
            raise ValueError("The higher interval must be a multiple of the lower interval.")
        return slow_minutes // fast_minutes

    def get_rounded_time(self, timestamp, interval):
        """将时间戳向下取整到指定的时间单位"""
        minutes = self.get_interval_minutes(interval)
        total_minutes = timestamp.minute + timestamp.hour * 60
        rounded_total_minutes = (total_minutes // minutes) * minutes
        return timestamp.replace(hour=rounded_total_minutes // 60, minute=rounded_total_minutes % 60, second=0, microsecond=0)

    def animate(self, i):
        if self.current_index + 1 < len(self.minute_data):
            self.current_index += 1
        else:
            return

        # 获取当前时间和小时间单位数据
        current_time = self.minute_data['timestamp'].iloc[self.current_index]
        minute_df = self.minute_data.iloc[max(0, self.current_index - self.window * self.interval_factor):self.current_index]
        min_strategy_df = self.minute_data.iloc[max(0, self.current_index - 200):self.current_index]
        
        # 计算对应的高时间单位时间
        higher_time = self.get_rounded_time(current_time, self.slow_interval)
        higher_df = self.higher_data[self.higher_data['timestamp'] <= higher_time].tail(self.window)
        
        # 获取最新的200条大级别数据
        higher_strategy_df = self.higher_data.iloc[max(0, len(self.higher_data) - 200):]

        # 动态更新当前高时间单位K线
        if len(higher_df) > 0:
            last_higher = higher_df.iloc[-1].copy()
            last_higher['close'] = minute_df['close'].iloc[-1]
            self.accumulated_volume += float(minute_df['volume'].iloc[-1])  # 累积交易量，确保交易量是数值类型
            last_higher['volume'] = self.accumulated_volume
            higher_df = pd.concat([higher_df.iloc[:-1], pd.DataFrame([last_higher])])
        
        # 如果新的高时间单位K线开始了，重置累积交易量
        if self.previous_higher_time and self.previous_higher_time != higher_time:
            self.accumulated_volume = 0
        self.previous_higher_time = higher_time

        # 调用set_signals方法来设置信号
        self.set_signals(current_time, minute_df['close'].iloc[-1], min_strategy_df, higher_strategy_df)
        
        self.ax1.clear()
        self.ax2.clear()
        
        self.plot_candlestick(higher_df, self.ax1, 'Higher Time Frame Data')
        
        # 绘制当前窗口内的买卖信号
        window_start_time = higher_df['timestamp'].min()
        window_end_time = higher_df['timestamp'].max()

        buys = [signal for signal in self.buy_signals if window_start_time <= signal['timestamp'] <= window_end_time]
        sells = [signal for signal in self.sell_signals if window_start_time <= signal['timestamp'] <= window_end_time]

        for buy in buys:
            self.ax1.plot(buy['timestamp'], buy['price'], marker='^', color='g', markersize=10, label='Buy Signal')

        for sell in sells:
            self.ax1.plot(sell['timestamp'], sell['price'], marker='v', color='r', markersize=10, label='Sell Signal')

        self.ax1.legend()
        self.ax1.xaxis.set_visible(False)  # 隐藏第一个子图的 x 轴
        self.ax1.set_ylabel('Price')
        self.ax1.set_title('Real-time Trading Backtest - Combined Data')

        # 绘制交易量
        bar_width = self.get_bar_width()  # 使用自定义方法获取柱子的宽度
        self.ax2.bar(higher_df['timestamp'], higher_df['volume'], width=bar_width, color='blue', alpha=0.5)
        self.ax2.set_xlabel('Time')
        self.ax2.set_ylabel('Volume')

        # 旋转 x 轴标签
        plt.setp(self.ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        self.ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))


    def get_bar_width(self):
        """动态计算柱子的宽度以保持一致的视觉效果"""
        fast_minutes = self.get_interval_minutes(self.fast_interval)
        slow_minutes = self.get_interval_minutes(self.slow_interval)
        width_ratio = fast_minutes / slow_minutes
        return max(0.5, width_ratio * 0.5)  # 确保宽度不小于某个值，防止柱子过于窄

    def plot_candlestick(self, df, ax, label, alpha=1.0):
        for idx, row in df.iterrows():
            color = 'green' if row['close'] >= row['open'] else 'red'
            ax.plot([row['timestamp'], row['timestamp']], [row['low'], row['high']], color='black', alpha=alpha)
            ax.plot([row['timestamp'], row['timestamp']], [row['open'], row['close']], color=color, linewidth=5, alpha=alpha)
        ax.set_label(label)

    def run(self):
        self.animate(0)
        ani = animation.FuncAnimation(self.fig, self.animate, interval=self.speed)
        plt.show()

    def add_buy_signal(self, timestamp, price):
        self.buy_signals.append({'timestamp': timestamp, 'price': price})

    def add_sell_signal(self, timestamp, price):
        self.sell_signals.append({'timestamp': timestamp, 'price': price})

    def set_signals(self, current_time, current_price, min_strategy_df, higher_strategy_df):
        """使用当前最新的时间和价格来设置买入和卖出信号"""
        pass
        # 示例：基于某些条件添加买入和卖出信号
        # if some_buy_condition:
        #     self.add_buy_signal(current_time, current_price)
        # if some_sell_condition:
        #     self.add_sell_signal(current_time, current_price)
        
        