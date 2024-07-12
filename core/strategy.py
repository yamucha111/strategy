import pandas as pd
import talib

class Strategy:
    @staticmethod
    def calculate_topn_volume_averages(data, topn=3):
        """
        计算5条数据内的交易量平均值，并返回排名前三的交易量平均值。
        
        :param data: pd.DataFrame 包含 'volume' 列的数据。
        :return: 排名前三的交易量平均值。
        """
        if 'volume' not in data.columns:
            raise ValueError("Data must contain 'volume' column")

        if data.shape[0] < 5:
            raise ValueError("Data must contain at least 5 rows")

        # 计算交易量的EMA
        data['volume_ema'] = talib.EMA(data['volume'], timeperiod=5)

        # 取出计算后的EMA，去除NaN值
        volume_ema_values = data['volume_ema'].dropna().values

        # 对均值进行排序并返回前三名
        topn_averages = sorted(volume_ema_values, reverse=True)[:topn]
        return topn_averages

