from backtesting import Strategy
import pandas as pd
import pandas_ta as ta

class EMA_RSI_TrendFollowing(Strategy):
    ema_short_len = 20
    ema_long_len = 50
    rsi_len = 14
    roc_len = 3

    def init(self):
        self.ema_short = self.I(ta.ema, pd.Series(self.data.Close), self.ema_short_len)
        self.ema_long = self.I(ta.ema, pd.Series(self.data.Close), self.ema_long_len)
        self.rsi = self.I(ta.rsi, pd.Series(self.data.Close), self.rsi_len)
        self.roc = self.I(ta.roc, pd.Series(self.data.Close), self.roc_len)

    def next(self):
        price = self.data.Close[-1]
        tp = price * 1.03
        sl = price * 0.97

        if (
            self.data.Close[-1] > self.ema_short[-1]
            and self.data.Close[-1] > self.ema_long[-1]
            and 50 < self.rsi[-1] < 65
            and 0 < self.roc[-1] < 7
        ):
            self.buy(sl=sl, tp=tp, size=0.2)
