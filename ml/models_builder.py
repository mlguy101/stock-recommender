"""
1- Get Hopt calibration data
2- calibrate find-peak parameters based on backtest.py optimize
3- use these parameters to detect local maxima, minima
4- for each local minima / maxima , calculate the slope for next maxima / minima ( later idea)
5- this normalized slope presents the by which we recommend buy sell
6 - brainstorm limitations
7- create model to predict local maxima / minima from set of indicators and past data only (lagged indicators ?? )
8- iterate and backtest
9- test live with fake money
10  -think about adding risk factor , this will control buy sell based on the accuracy  / confidence of the prediction
"""
import logging
import plotly.express as px

import numpy as np
import yfinance
from backtesting import Backtest, backtesting
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from talipp.indicators import EMA

from ml.strategy_builder import StrategyBuilder, HyperParamLocalMinMaxStrategy


class LocalTurningMLStrategyBuilder(StrategyBuilder):
    """
    Strategy based on simply predicting local minima / maxima points as the best buy / sell positions
    """

    def __init__(self, ticker: str):
        self.ticker = ticker
        self.logger = logging.getLogger()

    def peak_finder_hopt(self, start: str, end: str, interval: str) -> dict:
        """

        :param end:
        :param start:
        :param interval:
        :return:
        """
        # FIXME Data Retrieval Reliability issue
        """
        <ticker>  No timezone found, symbol may be de-listed 
        https://github.com/ranaroussi/yfinance/issues/359 
        """
        logger = logging.getLogger()
        df = self.get_data(ticker=self.ticker, start=start, end=end, interval=interval)
        if df.shape[0] < 1:
            self.logger.error(f'Cannot download data for ticker {self.ticker}')
            return {'ticker': self.ticker, 'opt_stats': None}
        # TODO understand exclusive orders
        # must add to have valid number of trades and SQN
        bt = Backtest(data=df, strategy=HyperParamLocalMinMaxStrategy, commission=0.005, cash=10000,
                      exclusive_orders=True)
        # TODO take care of max obj
        # https://indextrader.com.au/van-tharps-sqn/
        # https://github.com/kernc/backtesting.py/blob/master/backtesting/backtesting.py#L1264

        opt_stats = bt.optimize(lookahead=list(np.arange(1, 5)), maximize='SQN')

        # logger.info(f'For ticker = {self.ticker}, the optimal stats is \n{opt_stats}\n '
        #             f'with optimal strategy \n{opt_stats._strategy}')
        # stats = bt.run(lookahead=opt_stats._strategy.lookahead, verbose=True, start=False)
        # fig = px.line(x=df.index, y=df['Adj Close'])
        # fig.show()

        return {'ticker': self.ticker, 'opt_stats': opt_stats}


class LocalTurningPointsModelBuilder:
    def __init__(self, ticker, target_ma_window, width, prominence, distance):
        self.ticker = ticker
        self.width = width
        self.prominence = prominence
        self.distance = distance
        self.target_ma_window = target_ma_window

    def generate_feature_matrix(self, start_datetime, end_datetime, interval):
        # def get ohlc data
        # generate signals  (X or features)

        # generate targets ( Y={1,0,-1} )
        # Signals are generate at t i.e. X(t) and predict Y(t+1)
        # focus first on daily level
        """

        :return:
        """
        """
        Features 
        1- EMA(K) : K = 3, 5, 7
        2- 
        2- 1st and 2nd Diff of EMA(K)
        """
        # get raw ohlc data
        df = yfinance.download(tickers=self.ticker, start=start_datetime, end=end_datetime, interval=interval)
        # Generate X in features
        features_mtx = {}
        # Generate EMA
        periods = [5, 10]
        for period in periods:
            tmp = EMA(period=period, input_values=df['Adj Close'].values)
            ema = [np.nan] * (period - 1)
            ema.extend(tmp)
            features_mtx[f'ema_{period}'] = ema

    # get Y

    def build_local_turning_point_prediction_model(self):
        # model must predict y along with confidence level in prediction
        pass
