# https://pypi.org/project/Backtesting/
import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from datetime import date, timedelta
from backtesting.test import SMA
import yfinance as yf
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from talipp.indicators import EMA


class SmaCross(Strategy):
    def init(self):
        price = self.data.Close
        self.ma1 = self.I(SMA, price, 10)
        self.ma2 = self.I(SMA, price, 20)

    def next(self):
        print(self.data.index._data[-1])
        if crossover(self.ma1, self.ma2):
            self.buy()
        elif crossover(self.ma2, self.ma1):
            self.sell()


def get_signals(n, local_maxima_idx, local_minima_idx):
    signals = np.zeros(n)
    for i in range(n):
        if i in local_maxima_idx:
            signals[i] = 1
        elif i in local_minima_idx:
            signals[i] = -1
        else:
            signals[i] = 0
    return signals


class HyperParamLocalMinMaxStrategy(Strategy):
    prominence = 0.3
    distance = 100

    def __init__(self, broker, data, params):
        super().__init__(broker, data, params)

    def init(self):
        dt_idx = self.data.df.index
        close_data = self.data.Close.data
        window = 10
        ema = np.array([np.nan] * (window - 1) + EMA(period=window, input_values=close_data).output_values)
        assert len(ema) == len(dt_idx)
        local_maxima_idx, _ = find_peaks(x=ema, prominence=self.prominence, distance=self.distance)
        local_minima_idx, _ = find_peaks(x=-ema, prominence=self.prominence, distance=self.distance)
        signals = get_signals(n=len(ema), local_maxima_idx=local_maxima_idx, local_minima_idx=local_minima_idx)
        self.signals_df = pd.DataFrame({'signals': signals})
        self.signals_df.index = dt_idx
        # plt.plot(ema)
        # plt.plot(local_maxima_idx, ema[local_maxima_idx], 'x')
        # plt.plot(local_minima_idx, ema[local_minima_idx], 'o')
        # plt.show()

    def next(self):
        dt = self.data.index._data[-1]
        signal = self.signals_df['signals'][dt]
        if signal == 1:
            self.sell()
        elif signal == -1:
            self.buy()


if __name__ == '__main__':
    sp500 = pd.read_csv('SP500.csv', sep=',')
    for symbol in sp500['Symbol']:
        print(f'Symbol = {symbol}')
        interval = '60m'
        end = date.today().isoformat()
        start = (date.today() - timedelta(days=180)).isoformat()
        df = yf.download(tickers=symbol, start=start, end=end, interval=interval)
        if df.shape[0] < 2:
            print(f'Cannot get data for symbol {symbol}')
            continue
        bt = Backtest(data=df, strategy=HyperParamLocalMinMaxStrategy, commission=0.005, cash=10_000)
        statsopt = bt.optimize(prominence=[0.1, 0.2, 0.4, 0.9], distance=[10, 20, 30, 40, 50, 60, 70, 100, 150, 200],
                               maximize='Equity Final [$]')
        print(statsopt._strategy)
        print('#################################################################')
