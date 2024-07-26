from datetime import datetime, timezone
import pandas as pd
import random

class Utils:
    
    @staticmethod
    def timestamp_to_date_string(timestamp, format='%Y-%m-%d %H:%M:%S', tz=timezone.utc):
        """
        时间戳转日期字符串
        """
        dt = datetime.fromtimestamp(timestamp, tz=tz)
        return dt.strftime(format)
    
    @staticmethod
    def date_string_to_timestamp(date_string, format='%Y-%m-%d %H:%M:%S'):
        """
        日期字符串转时间戳
        """
        dt = datetime.strptime(date_string, format)
        dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())

    @staticmethod
    def date_string_to_date(date_string, format='%Y-%m-%d %H:%M:%S', tzinfo=timezone.utc):
        """
        日期字符串转日期对象
        """
        dt = datetime.strptime(date_string, format)
        return dt.replace(tzinfo=tzinfo)

    @staticmethod
    def date_to_date_string(date_obj, format='%Y-%m-%d %H:%M:%S'):
        """
        日期对象转日期字符串
        """
        return date_obj.strftime(format)

    @staticmethod
    def date_to_timestamp(date_obj):
        """
        日期对象转时间戳
        """
        return int(date_obj.timestamp())

    @staticmethod
    def timestamp_to_date(timestamp):
        """
        时间戳转日期对象
        """
        return datetime.fromtimestamp(timestamp, tz=timezone.utc)

    @staticmethod
    def generate_random_number(level):
        """
        Generate a random float number based on the level.
        The higher the level, the higher the probability of generating a larger number.

        Args:
            level (int): The level determining the range of the random number.

        Returns:
            float: A random number greater than 0.
        """
        exponent = 1 + level / 10  # Adjust the exponent based on the level

        random_number = random.expovariate(1 / exponent)
        
        return random_number
    
    @staticmethod
    def str_to_numeric(combined_df):
        # 在读取数据后，确保数值列的类型正确
        combined_df['open'] = pd.to_numeric(combined_df['open'], errors='coerce')
        combined_df['high'] = pd.to_numeric(combined_df['high'], errors='coerce')
        combined_df['low'] = pd.to_numeric(combined_df['low'], errors='coerce')
        combined_df['close'] = pd.to_numeric(combined_df['close'], errors='coerce')
        combined_df['volume'] = pd.to_numeric(combined_df['volume'], errors='coerce')
        return combined_df