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

    def init(self):
        dt_idx = self.data.df.index
        close_data = self.data.Close.data
        window = 10
        ema = np.array([np.nan] * (window - 1) + EMA(period=window, input_values=close_data).output_values)
        assert len(ema) == len(dt_idx)
        prominence = 0.8
        distance = 10
        local_maxima_idx, _ = find_peaks(x=ema, prominence=prominence, distance=distance)
        local_minima_idx, _ = find_peaks(x=-ema, prominence=prominence, distance=distance)
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
    # bt = Backtest(GOOG, SmaCross, commission=.002,
    #               exclusive_orders=True)
    ticker = 'AAPL'
    interval = '60m'
    end = date.today().isoformat()
    start = (date.today() - timedelta(days=180)).isoformat()
    df = yf.download(tickers=ticker, start=start, end=end, interval=interval)
    bt = Backtest(data=df, strategy=HyperParamLocalMinMaxStrategy, commission=0.002, cash=10_000)
    statsopt = bt.optimize(prominence=[0.1, 0.2, 0.4, 0.9], maximize='Equity Final [$]')
    print(statsopt)
    print('*****')
    print(statsopt._strategy)
