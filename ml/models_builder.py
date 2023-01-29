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
from enum import Enum
from typing import List, Any

import pandas as pd
import plotly.express as px

import numpy as np
import yfinance
from backtesting import Backtest
from peakdetect import peakdetect
from sklearn.metrics import accuracy_score
from talipp.indicators import EMA
from xgboost import XGBClassifier


class TurningPoint(Enum):
    LOCAL_MAXIMA = 2
    LOCAL_MINIMA = 0
    NEITHER = 1

class TurningPointsModelBuilder:
    ADJ_CLOSE_COLNAME = 'Adj Close'

    def __init__(self, ticker, target_ma_window, lookahead, delta):
        self.lookahead = lookahead
        self.target_ma_window = target_ma_window
        self.logger = logging.getLogger()
        self.ticker = ticker
        self.delta = delta

    @staticmethod
    def generate_features_from_prices(ticker: str, prices: pd.Series, ema_periods: List[int],
                                      diffs: List[int]) -> pd.DataFrame:
        """

        :param ticker:
        :param prices:
        :param ema_periods:
        :param diffs:
        :return:
        """

        # TODO
        """
        calculate 
        ema
        1st and 2nd order diff (diff of diff) , fix this
        lags of features
        """
        ema_dict = {}
        logger = logging.getLogger()
        nan_count = sum(np.isnan(prices))
        logger.info(f'ticker : {ticker},nan count = {nan_count}')
        for ema_period in ema_periods:
            ema_dict[f'ema_{ema_period}'] = [np.nan] * (ema_period - 1)
            tmp = EMA(period=ema_period, input_values=prices.values)
            ema_dict[f'ema_{ema_period}'].extend(tmp)
        ema_df = pd.DataFrame(ema_dict)
        ema_df.index = prices.index
        diff_dfs = [ema_df]
        for diff in diffs:
            diff_df = ema_df.diff(periods=diff)
            diff_df.index = prices.index
            rename_params = {}
            for col in diff_df.columns:
                rename_params[col] = f'{col}_diff_{diff}'
            diff_df.rename(columns=rename_params,inplace=True)
            diff_dfs.append(diff_df)
        features_df = pd.concat(objs=diff_dfs)
        return features_df

    def train_model(self, df):
        # def get ohlc data
        # generate signals  (X or features

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
        TurningPointsModelBuilder.generate_features_from_prices(ticker=self.ticker,
                                                                prices=df[TurningPointsModelBuilder.ADJ_CLOSE_COLNAME],
                                                                ema_periods=[5, 10], diffs=[1, 2])
        tmp = EMA(period=5, input_values=df['Adj Close'].values)
        ema = [np.nan] * (period - 1)
        ema.extend(tmp)
        feature_mtx[f'ema_{period}'] = pd.Series(ema, index=df.index)  # get Y

        peak = peakdetect(y_axis=df['Adj Close'], x_axis=df.index, lookahead=self.lookahead, delta=self.delta)
        assert len(peak) == 2, "peak array must be 2D"

        local_maxima = np.array(peak[0])
        local_minima = np.array(peak[1])
        self.logger.info(f'ticker = {self.ticker} number of local maxima = {len(local_maxima)}')
        self.logger.info(f'ticker = {self.ticker} number of local minima = {len(local_minima)}')
        Y = TurningPointsModelBuilder.get_signals(index=list(df.index), local_minima_idx=list(local_minima[:, 0]),
                                                      local_maxima_idx=list(local_maxima[:, 0]))
        Y_num = np.array(list(map(lambda x: x.value.real, Y)))

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
        N_train = int(round(0.2 * N))

        X_train = feature_mtx_df_reduced.iloc[:N_train]
        X_test = feature_mtx_df_reduced.iloc[N_train:]
        Y_num_train = Y_num[:N_train]
        Y_num_test = Y_num[N_train:]

        model = XGBClassifier()
        bst = model.fit(X_train, Y_num_train)
        y_pred = np.array(model.predict(X_test))
        predictions = np.array([round(value) for value in y_pred])
        peak_idx = [y in [TurningPoint.LOCAL_MAXIMA.value, TurningPoint.LOCAL_MINIMA.value] for y in Y_num_test]
        y_test_peaks = Y_num_test[peak_idx]
        y_pred_peaks = predictions[peak_idx]
        assert len(y_pred_peaks) == len(y_pred_peaks)
        peaks_accuracy = accuracy_score(y_true=y_test_peaks, y_pred=y_pred_peaks) if len(y_test_peaks) > 0 else 0.0
        scores = bst.feature_importances_
        scores_df = pd.DataFrame({'features': X_train.columns, 'score': scores})
        scores_df.sort_values(by='score', ascending=False, inplace=True)
        return bst, peaks_accuracy

    @staticmethod
    def get_signals(index: List[Any], local_maxima_idx: List[Any], local_minima_idx: List[Any]) -> pd.Series:
        """

        :param n:
        :param local_maxima_idx:
        :param local_minima_idx:
        :return:
        """
        signals = pd.Series([None] * len(index), index=index)
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
