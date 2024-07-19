import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from core.strategy import Strategy
import talib
pd.set_option('mode.chained_assignment', None)

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
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(16, 8), gridspec_kw={'height_ratios': [2, 1]})
        self.buy_signals = []
        self.sell_signals = []
        self.accumulated_volume = 0  # 用于累积当前高时间单位K线的交易量
        self.previous_higher_time = None  # 用于记录上一个高时间单位时间戳
        self.macd_values = None  # 存储初始的MACD计算值
        self.local_maxima = []  # 全局记录局部最高点
        self.local_minima = []  # 全局记录局部最低点
        self.searching_for_max = True  # 初始状态为寻找最高点
        self.last_extreme_timestamp = None  # 记录最后一个极值点的时间戳
        self.confirmation_window = 5  # 确认极值点的窗口大小
        self.volume_breakout_points = {'up': [], 'down': []}  # 记录交易量突破点
        self.last_volume_timestamp = None  # 记录上次处理的最后一个数据点的时间戳
        self.init_legend()
        
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

    def init_legend(self):
        """初始化图例"""
        self.ax1.plot([], [], marker='^', color='g', markersize=10, label='Buy Signal')
        self.ax1.plot([], [], marker='v', color='r', markersize=10, label='Sell Signal')
        self.ax1.legend()

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
        higher_strategy_df = self.higher_data[self.higher_data['timestamp'] <= higher_time].tail(200).copy()

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
        higher_strategy_df.iloc[-1] = higher_df.iloc[-1]
        self.set_signals(current_time, minute_df['close'].iloc[-1], min_strategy_df, higher_strategy_df)
        
        self.ax1.clear()
        self.ax2.clear()
        # self.ax3.clear()
        
        self.plot_candlestick(higher_df, self.ax1, 'Higher Time Frame Data')
        
        # 绘制布林带
        upperband, middleband, lowerband = talib.BBANDS(higher_df['close'], timeperiod=20)
        self.ax1.plot(higher_df['timestamp'], upperband, label='Upper Band', color='blue', linestyle='--')
        self.ax1.plot(higher_df['timestamp'], middleband, label='Middle Band', color='black', linestyle='--')
        self.ax1.plot(higher_df['timestamp'], lowerband, label='Lower Band', color='blue', linestyle='--')
        
        # 绘制MA120
        ma120 = talib.SMA(higher_df['close'], timeperiod=120)
        self.ax1.plot(higher_df['timestamp'], ma120, label='MA120', color='purple')
        
        # 绘制当前窗口内的买卖信号
        window_start_time = higher_df['timestamp'].min()
        window_end_time = higher_df['timestamp'].max()

        buys = [signal for signal in self.buy_signals if window_start_time <= signal['timestamp'] <= window_end_time]
        sells = [signal for signal in self.sell_signals if window_start_time <= signal['timestamp'] <= window_end_time]

        for buy in buys:
            self.ax1.plot(buy['timestamp'], buy['price'], marker='^', color='blue', markersize=10)

        for sell in sells:
            self.ax1.plot(sell['timestamp'], sell['price'], marker='v', color='black', markersize=10)
            
        # 更新局部极值点
        self.update_local_extremes(higher_df)

        # 绘制局部极值点
        self.plot_local_extremes(higher_df)
        
        # 检查支撑位和压力位
        self.check_support_resistance(higher_df)

        # 绘制支撑位和压力位
        self.plot_support_resistance(higher_df)

        self.ax1.legend()
        self.ax1.xaxis.set_visible(False)  # 隐藏第一个子图的 x 轴
        self.ax1.set_ylabel('Price')
        self.ax1.set_title('Real-time Trading Backtest - Combined Data')

        # 绘制交易量
        bar_width = self.get_bar_width()  # 使用自定义方法获取柱子的宽度
        self.ax2.bar(higher_df['timestamp'], higher_df['volume'], width=bar_width, color='blue', alpha=0.5)
        self.ax2.set_xlabel('Time')
        self.ax2.set_ylabel('Volume')

        # # 动态计算并更新MACD
        # macd, signal, hist = talib.MACD(higher_df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        # if self.macd_values is None:
        #     self.macd_values = {
        #         'timestamp': higher_df['timestamp'].apply(mdates.date2num).values,
        #         'macd': macd,
        #         'signal': signal,
        #         'hist': hist
        #     }
        # else:
        #     self.macd_values['timestamp'] = np.append(self.macd_values['timestamp'][-self.current_index:], mdates.date2num(higher_df['timestamp'].iloc[-1]))
        #     self.macd_values['macd'] = np.append(self.macd_values['macd'][-self.current_index:], macd.iloc[-1])
        #     self.macd_values['signal'] = np.append(self.macd_values['signal'][-self.current_index:], signal.iloc[-1])
        #     self.macd_values['hist'] = np.append(self.macd_values['hist'][-self.current_index:], hist.iloc[-1])

        # self.ax3.plot(self.macd_values['timestamp'], self.macd_values['macd'], label='MACD', color='blue')
        # self.ax3.plot(self.macd_values['timestamp'], self.macd_values['signal'], label='Signal', color='red')
        # self.ax3.bar(self.macd_values['timestamp'], self.macd_values['hist'], label='Hist', color='gray', alpha=0.5)

        # self.ax3.legend()
        # self.ax3.set_ylabel('MACD')
        # self.ax3.set_xlabel('Time')

        # 旋转 x 轴标签
        plt.setp(self.ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        self.ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        # plt.setp(self.ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
        # self.ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    
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
        
    def update_local_extremes(self, df, min_interval=5):
        """更新和绘制局部最高点和最低点，且最高点和最低点之间需要有指定数量的有效K线间隔"""
        if df.shape[0] < min_interval:
            return
        
        if self.last_extreme_timestamp:
            df = df[df['timestamp'] > self.last_extreme_timestamp].reset_index(drop=True)
            if df.empty:
                return
            start_index = 0
        else:
            start_index = 0
            
        df_len = df.shape[0]
        for i in range(start_index, df_len - min_interval):
            if i >= len(df):
                break
            
            if self.searching_for_max:
                # 寻找局部最高点
                window = df.iloc[max(0, i - min_interval):min(df_len, i + min_interval + 1)]
                if df['high'].iloc[i] == window['high'].max():
                    potential_max = {'timestamp': df['timestamp'].iloc[i], 'price': df['high'].iloc[i]}
                    max_confirmed = False

                    # 在接下来的K线内寻找低于该最高点的K线，并开始计数
                    for j in range(i + 1, df_len - min_interval):
                        if df['low'].iloc[j] < df['low'].iloc[i]:
                            # 找到低点，开始计数接下来的5条K线
                            max_confirmed = True
                            for k in range(j + 1, min(j + 1 + min_interval, len(df))):
                                if df['high'].iloc[k] > df['high'].iloc[i]:
                                    max_confirmed = False
                                    break
                            if max_confirmed:
                                i = j  # 从找到的低点位置开始继续寻找
                                break

                    if max_confirmed:
                        self.local_maxima.append(potential_max)
                        self.searching_for_max = False  # 转换为寻找最低点
                        self.last_extreme_timestamp = df['timestamp'].iloc[i]
                        break

            else:
                # 寻找局部最低点
                window = df.iloc[max(0, i - min_interval):min(df_len, i + min_interval + 1)]
                if df['low'].iloc[i] == window['low'].min():
                    potential_min = {'timestamp': df['timestamp'].iloc[i], 'price': df['low'].iloc[i]}
                    min_confirmed = False

                    # 在接下来的K线内寻找高于该最低点的K线，并开始计数
                    for j in range(i + 1, df_len - min_interval):
                        if df['high'].iloc[j] > df['high'].iloc[i]:
                            # 找到高点，开始计数接下来的5条K线
                            min_confirmed = True
                            for k in range(j + 1, min(j + 1 + min_interval, len(df))):
                                if df['low'].iloc[k] < df['low'].iloc[i]:
                                    min_confirmed = False
                                    break
                            if min_confirmed:
                                i = j  # 从找到的高点位置开始继续寻找
                                break

                    if min_confirmed:
                        self.local_minima.append(potential_min)
                        self.searching_for_max = True  # 转换为寻找最高点
                        self.last_extreme_timestamp = df['timestamp'].iloc[i]
                        break

    def plot_local_extremes(self, df):
        """绘制局部最高点和最低点"""
        window_start_time = df['timestamp'].min()
        window_end_time = df['timestamp'].max()

        filtered_maxima = [max_point for max_point in self.local_maxima if window_start_time <= max_point['timestamp'] <= window_end_time]
        for max_point in filtered_maxima:
            timestamp = max_point['timestamp']
            price = max_point['price']
            self.ax1.plot(timestamp, price, marker='o', color='blue', markersize=5)

        filtered_minima = [min_point for min_point in self.local_minima if window_start_time <= min_point['timestamp'] <= window_end_time]
        for min_point in filtered_minima:
            timestamp = min_point['timestamp']
            price = min_point['price']
            self.ax1.plot(timestamp, price, marker='o', color='orange', markersize=5)
            
    def check_support_resistance(self, df):
        """检查并保存交易量突破点，并标记在图表上"""
        higher_strategy_df_120 = df.iloc[-120:]
        self.top3_volume = Strategy.calculate_topn_volume_averages(higher_strategy_df_120)

        if self.last_volume_timestamp is not None:
            df = df[df['timestamp'] > self.last_volume_timestamp].reset_index(drop=True)
            if df.empty:
                return

        for i in range(len(df)):
            if df['volume'].iloc[i] > self.top3_volume[0]:
                if df['close'].iloc[i] > df['open'].iloc[i]:
                    self.volume_breakout_points['up'].append((df['timestamp'].iloc[i], df['high'].iloc[i]))
                else:
                    self.volume_breakout_points['down'].append((df['timestamp'].iloc[i], df['low'].iloc[i]))
                self.last_volume_timestamp = df['timestamp'].iloc[i]

    def plot_support_resistance(self, df):
        """绘制当前窗口内的交易量突破点"""
        window_start_time = df['timestamp'].min()
        window_end_time = df['timestamp'].max()

        filtered_up_points = [point for point in self.volume_breakout_points['up'] if window_start_time <= point[0] <= window_end_time]
        filtered_down_points = [point for point in self.volume_breakout_points['down'] if window_start_time <= point[0] <= window_end_time]

        for point in filtered_up_points:
            self.ax1.plot(point[0], point[1], '_', markersize=15)

        for point in filtered_down_points:
            self.ax1.plot(point[0], point[1], '_', markersize=15)

    def run(self):
        self.animate(0)
        ani = animation.FuncAnimation(self.fig, self.animate, interval=self.speed)
        plt.show()

    def add_buy_signal(self, timestamp, price):
        self.buy_signals.append({'timestamp': timestamp, 'price': price})
        self.buy_signals = self.buy_signals[-self.window:]

    def add_sell_signal(self, timestamp, price):
        self.sell_signals.append({'timestamp': timestamp, 'price': price})
        self.sell_signals = self.sell_signals[-self.window:]

    def set_signals(self, current_time, current_price, min_strategy_df, higher_strategy_df):
        """使用当前最新的时间和价格来设置买入和卖出信号"""
        higher_strategy_df_120 = higher_strategy_df.iloc[-120:]
        top3_volume = Strategy.calculate_topn_volume_averages(higher_strategy_df_120)
        ema5_top1_volume = top3_volume[0]
        ema5s = talib.EMA(higher_strategy_df['close'], timeperiod=5)
        last_ema5 = ema5s.iloc[-1]
        pre_ema5 = ema5s.iloc[-2]
        ema120s = talib.EMA(higher_strategy_df['close'], timeperiod=120)
        last_ema120 = ema120s.iloc[-1]
        pre_ema120 = ema120s.iloc[-2]
        last_ema140 = talib.EMA(higher_strategy_df['close'], timeperiod=140).iloc[-1]
        last_ema160 = talib.EMA(higher_strategy_df['close'], timeperiod=160).iloc[-1]
        ma120s = talib.SMA(higher_strategy_df['close'], timeperiod=120)
        last_ma120 = ma120s.iloc[-1]
        pre_ma120 = ma120s.iloc[-2]
        
        min_macd, min_signal, min_his = talib.MACD(min_strategy_df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        min_last_signal = min_signal.iloc[-1]
        min_pre_signal = min_signal.iloc[-2]
        
        higher_macd, higher_signal, higher_his = talib.MACD(higher_strategy_df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        higher_last_signal = higher_his.iloc[-1]
        higher_pre_signal = higher_his.iloc[-2]
        
        upperband, middleband, lowerband = talib.BBANDS(higher_strategy_df['close'], timeperiod=20)
        
        up_trend_ema120 = last_ema120 > pre_ema120 and last_ema120 > pre_ma120
        down_trend_ema120 = last_ema120 < pre_ema120 and last_ema120 < pre_ma120
        
        up_trend_ema5 = last_ema5 > pre_ema5 and last_ema5 > pre_ma120
        
        if not np.isnan([last_ema120, last_ema140, last_ema160, last_ma120]).any():
            max_standard_price = max(last_ema120, last_ema140, last_ema160, last_ma120)
            min_standard_price = min(last_ema120, last_ema140, last_ema160, last_ma120)
            
            if current_price > max_standard_price and higher_last_signal < 0 and current_price < middleband.iloc[-1]:
                if min_pre_signal < 0 and min_last_signal >= 0:
                    if current_price < middleband.iloc[-1]:
                        self.add_buy_signal(current_time, current_price)
            
            if current_price < min_standard_price and higher_last_signal > 0:
                if min_pre_signal > 0 and min_last_signal <= 0:
                    if current_price > middleband.iloc[-1]:
                        self.add_sell_signal(current_time, current_price)
