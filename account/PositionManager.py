

class PositionManager:
    def __init__(self, initial_balance, pair, leverage, slippage, fee_rate):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.pair = pair
        self.leverage = leverage
        self.slippage = slippage
        self.fee_rate = fee_rate
        self.positions = []  # 存储持仓信息
        self.trades = []  # 存储交易记录

    def open_position(self, timestamp, price, direction, size):
        position = {
            'timestamp': timestamp,
            'price': price,
            'direction': direction,
            'size': size,
            'entry_fee': price * size * self.fee_rate,
            'exit_fee': 0,
            'exit_price': 0,
            'pnl': 0
        }
        self.positions.append(position)
        self.balance -= position['entry_fee']  # 扣除开仓手续费

    def close_position(self, timestamp, price):
        for position in self.positions:
            if position['exit_price'] == 0:  # 只关闭未平仓的持仓
                position['exit_price'] = price
                position['exit_fee'] = price * position['size'] * self.fee_rate
                if position['direction'] == 'long':
                    position['pnl'] = (price - position['price']) * position['size'] * self.leverage - position['exit_fee']
                else:
                    position['pnl'] = (position['price'] - price) * position['size'] * self.leverage - position['exit_fee']
                self.balance += position['pnl'] - position['exit_fee']  # 更新账户余额
                self.trades.append(position)
        self.positions = [p for p in self.positions if p['exit_price'] == 0]  # 移除已平仓的持仓

    def get_balance(self):
        return self.balance

    def get_trades(self):
        return self.trades

    def reset(self):
        self.balance = self.initial_balance
        self.positions = []
        self.trades = []