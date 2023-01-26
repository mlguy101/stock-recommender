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

import pandas as pd
import plotly.express as px

import numpy as np
import yfinance
from backtesting import Backtest
from peakdetect import peakdetect
from sklearn.metrics import accuracy_score
from talipp.indicators import EMA
from xgboost import XGBClassifier
from ml.strategy_builder import StrategyBuilder, HyperParamLocalMinMaxStrategy


class TurningModelBuilder(StrategyBuilder):
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


class TurningPointsModelBuilder:
    def __init__(self, target_ma_window, lookahead):
        self.lookahead = lookahead
        self.target_ma_window = target_ma_window

    def train_model(self, df):
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

        # Generate X in features
        feature_mtx = {}
        # Generate EMA
        period = 5
        tmp = EMA(period=5, input_values=df['Adj Close'].values)
        ema = [np.nan] * (period - 1)
        ema.extend(tmp)
        feature_mtx[f'ema_{period}'] = pd.Series(ema, index=df.index)  # get Y

        peak = peakdetect(y_axis=df['Adj Close'], x_axis=df.index, lookahead=self.lookahead)
        assert len(peak) == 2, "peak array must be 2D"
        local_maxima = np.array(peak[0])
        local_minima = np.array(peak[1])
        Y = HyperParamLocalMinMaxStrategy.get_signals(index=list(df.index), local_minima_idx=list(local_minima[:, 0]),
                                                      local_maxima_idx=list(local_maxima[:, 0]))
        Y_num = list(map(lambda x: x.value.real, Y))
        X_cols = ['ema_5']
        feature_mtx['Y'] = Y
        feature_mtx['Datetime'] = df.index.values
        feature_mtx_df = pd.DataFrame(feature_mtx)
        diff1 = feature_mtx_df[X_cols].diff(periods=1)
        diff2 = feature_mtx_df[X_cols].diff(periods=2)
        feature_mtx_df = pd.merge(left=feature_mtx_df, right=diff1, left_index=True, right_index=True,
                                  suffixes=('', '_diff1'))
        feature_mtx_df = pd.merge(left=feature_mtx_df, right=diff2, left_index=True, right_index=True,
                                  suffixes=('', '_diff2'))

        feature_mtx_df_reduced = feature_mtx_df[['ema_5', 'ema_5_diff1', 'ema_5_diff2']]
        N = feature_mtx_df_reduced.shape[0]
        N_train = int(round(0.8 * N))

        X_train = feature_mtx_df_reduced.iloc[:N_train]
        X_test = feature_mtx_df_reduced.iloc[N_train:]
        Y_num_train = Y_num[:N_train]
        Y_num_test = Y_num[N_train:]

        model = XGBClassifier()
        bst = model.fit(X_train, Y_num_train)
        y_pred = model.predict(X_test)
        predictions = [round(value) for value in y_pred]
        accuracy = accuracy_score(Y_num_test, predictions)
        scores = bst.feature_importances_
        scores_df = pd.DataFrame({'features': X_train.columns, 'score': scores})
        scores_df.sort_values(by='score', ascending=False, inplace=True)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))
        return bst
