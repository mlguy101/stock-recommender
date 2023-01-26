import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Iterable, List, Any

import numpy as np
import pandas as pd
import yfinance
from backtesting import Strategy
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from talipp.indicators import EMA, SMA
from peakdetect import peakdetect


class TurningPoint(Enum):
    LOCAL_MAXIMA = 2
    LOCAL_MINIMA = 0
    NEITHER = 1


class StrategyBuilder(ABC):
    @staticmethod
    def get_data(ticker: str, start: str, end: str, interval: str) -> pd.DataFrame:
        logger = logging.getLogger()
        logger.info(f'Getting data for {ticker} from yahoo-finance : start = {start},end = {end},interval = {interval}')
        df = yfinance.download(tickers=ticker, start=start, end=end, interval=interval)
        return df


class HyperParamLocalMinMaxStrategy(Strategy):
    lookahead = 3

    def __init__(self, broker, data, params):
        super().__init__(broker, data, params)

    def init(self):

        ma_window = 3
        dt_idx = self.data.df.index
        close_data = self.data.Close.data
        ema = np.array(
            [np.nan] * (ma_window - 1) + EMA(period=ma_window, input_values=close_data).output_values)
        assert len(ema) == len(dt_idx)
        # # https://stackoverflow.com/questions/1713335/peak-finding-algorithm-for-python-scipy
        # # https://stackoverflow.com/a/64588944/5937273
        peaks = peakdetect(y_axis=ema.data, lookahead=self.lookahead)
        local_maxima = np.array(peaks[0])
        local_minima = np.array(peaks[1])
        signals = self.get_signals(n=len(ema), local_maxima_idx=local_maxima[:, 0],
                                   local_minima_idx=local_minima[:, 0])
        self.signals_df = pd.DataFrame({'signals': signals})
        self.signals_df.index = dt_idx

    def next(self):
        dt = self.data.index._data[-1]
        signal = self.signals_df['signals'][dt]
        if signal == TurningPoint.LOCAL_MAXIMA:
            self.sell()
            # n1 = len(self.trades)
            # n2 = len(self.closed_trades)
            # print(f'sell @ dt = {dt}, close = {self.data.Close[-1]}')
        elif signal == TurningPoint.LOCAL_MINIMA:
            self.buy()
            # print(f'buy @ dt = {dt}, close = {self.data.Close[-1]}')

    # def plot(self, ticker):
    #     peaks = peakdetect(self.ema, lookahead=self.lookahead)
    #     # Lookahead is the distance to look ahead from a peak to determine if it is the actual peak.
    #     # Change lookahead as necessary
    #     higherPeaks = np.array(peaks[0])
    #     lowerPeaks = np.array(peaks[1])
    #     plt.plot(self.ema)
    #     plt.plot(higherPeaks[:, 0], higherPeaks[:, 1], 'ro')
    #     plt.plot(lowerPeaks[:, 0], lowerPeaks[:, 1], 'ko')
    #     plt.savefig(f'plots/plot_{ticker}.jpg')
    #     plt.clf()

    @staticmethod
    def get_signals(index: List[Any], local_maxima_idx: List[Any], local_minima_idx: List[Any])->pd.Series:
        """

        :param n:
        :param local_maxima_idx:
        :param local_minima_idx:
        :return:
        """
        signals = pd.Series([None] * len(index),index=index)
        if len(local_maxima_idx) > 0 and len(local_minima_idx) > 0 and local_maxima_idx[0] < local_minima_idx[0]:
            local_maxima_idx = local_maxima_idx[1:]
        if len(local_maxima_idx) > 0 and len(local_minima_idx) > 0 and local_minima_idx[-1] > local_maxima_idx[-1]:
            local_minima_idx = local_minima_idx[:-1]
        for i in index:
            if i in local_maxima_idx:
                signals[i] = TurningPoint.LOCAL_MAXIMA
            elif i in local_minima_idx:
                signals[i] = TurningPoint.LOCAL_MINIMA
            else:
                signals[i] = TurningPoint.NEITHER
        return signals
