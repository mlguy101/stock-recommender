import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance
from backtesting import Strategy
from scipy.signal import find_peaks
from talipp.indicators import EMA


class TurningPoint(Enum):
    LOCAL_MAXIMA = 1
    LOCAL_MINIMA = -1
    NEITHER = 0


class StrategyBuilder(ABC):
    @staticmethod
    def get_data(ticker: str, start: str, end: str, interval: str) -> pd.DataFrame:
        logger = logging.getLogger()
        logger.info(f'Getting data for {ticker} from yahoo-finance : start = {start},end = {end},interval = {interval}')
        df = yfinance.download(tickers=ticker, start=start, end=end, interval=interval)
        return df


class HyperParamLocalMinMaxStrategy(Strategy):
    prominence = 0.3
    distance = 100
    width = 10
    ma_window = 5

    def __init__(self, broker, data, params):
        super().__init__(broker, data, params)
        self.signals_df = None

    def init(self):
        dt_idx = self.data.df.index
        close_data = self.data.Close.data
        ema = np.array(
            [np.nan] * (self.ma_window - 1) + EMA(period=self.ma_window, input_values=close_data).output_values)
        assert len(ema) == len(dt_idx)
        local_maxima_idx, _ = find_peaks(x=ema, prominence=self.prominence, distance=self.distance, width=self.width)
        local_minima_idx, _ = find_peaks(x=-ema, prominence=self.prominence, distance=self.distance, width=self.width)
        signals = HyperParamLocalMinMaxStrategy.get_signals(n=len(ema), local_maxima_idx=local_maxima_idx,
                                                            local_minima_idx=local_minima_idx)
        self.signals_df = pd.DataFrame({'signals': signals})
        self.signals_df.index = dt_idx

    def next(self):
        dt = self.data.index._data[-1]
        signal = self.signals_df['signals'][dt]
        if signal == TurningPoint.LOCAL_MAXIMA:
            self.sell()
        elif signal == TurningPoint.LOCAL_MINIMA:
            self.buy()

    @staticmethod
    def get_signals(n: int, local_maxima_idx: Iterable[int], local_minima_idx: Iterable[int]):
        """

        :param n:
        :param local_maxima_idx:
        :param local_minima_idx:
        :return:
        """
        signals = [None] * n
        for i in range(n):
            if i in local_maxima_idx:
                signals[i] = TurningPoint.LOCAL_MAXIMA
            elif i in local_minima_idx:
                signals[i] = TurningPoint.LOCAL_MINIMA
            else:
                signals[i] = TurningPoint.NEITHER
        return signals
