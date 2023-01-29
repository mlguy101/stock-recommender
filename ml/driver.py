import logging

import pandas as pd
import requests
import yfinance
from datetime import date, timedelta

from backtesting import Backtest, Strategy
from talipp.indicators import EMA

from ml.models_builder import TurningPoint, TurningPointsModelBuilder


class MLTurningPointStrategy(Strategy):
    boost_model = None
    features_ema_window = 5
    min_data_size_ema_windows_diff = 10

    def init(self):
        pass

    def next(self):
        ema = [None] * (self.features_ema_window - 1)
        tmp = EMA(period=self.features_ema_window, input_values=list(self.data.Close))
        data_size = len(self.data.Close)

        if (data_size - self.features_ema_window) > self.min_data_size_ema_windows_diff:
            ema.extend(tmp)
            feature_mtx_dict = {f'ema_{self.features_ema_window}': ema}
            feature_mtx_df = pd.DataFrame(data=feature_mtx_dict)
            feature_mtx_df.index = self.data.index._data
            diff1 = feature_mtx_df.diff(periods=1)
            diff2 = feature_mtx_df.diff(periods=2)
            feature_mtx_df = pd.merge(left=feature_mtx_df, right=diff1, left_index=True,
                                      right_index=True, suffixes=('', '_diff1'))
            feature_mtx_df = pd.merge(left=feature_mtx_df, right=diff2, left_index=True,
                                      right_index=True, suffixes=('', '_diff2'))
            last_record = feature_mtx_df.iloc[-1, :].to_frame().T
            y_pred = boost_model.predict(last_record)
            y_pred_round = [round(y_hat) for y_hat in y_pred][0]
            if y_pred_round == TurningPoint.LOCAL_MINIMA.value:
                self.buy()
            elif y_pred_round == TurningPoint.LOCAL_MAXIMA.value:
                self.sell()




# TODO
"""
1. Measure Accuracy for local minima and maxima only (not normal points)
2. Measure precision recall for local_maxima and local_minima , separately
3. add one condition for buy , strong uptrend
4. buy with fraction of cash (10%) 
5. deeper in peakdetect algo 
"""
if __name__ == '__main__':
    # params #
    features_ema_window = 5
    target_ema_window = 3
    look_ahead = 2
    delta = 0.2
    end_day_test = date.today()
    start_day_test = end_day_test - timedelta(120)  # test last week
    end_day_train = start_day_test - timedelta(1)
    start_day_train = end_day_train - timedelta(150)

    n_tickers = 10
    commission = 0.0005
    start_cash = 10000
    min_data = 5
    #####################
    logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%Y-%m-%d:%H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger()
    logger.info(f'Getting SP00 tickers')
    sp500_df = pd.read_html(requests.get('https://www.slickcharts.com/sp500',
                                         headers={'User-agent': 'Mozilla/5.0'}).text)[0]
    sp500_df.sort_values(by='Weight', inplace=True, ascending=False)
    logger.info(f'Start train date = {start_day_train}')
    logger.info(f'End train date = {end_day_train}')
    logger.info(f'Start test date = {start_day_test}')
    logger.info(f'End test date = {end_day_test}')
    ##
    for ticker in sp500_df['Symbol'].values[:n_tickers]:
        train_df = yfinance.download(tickers=ticker, start=start_day_train.isoformat(), end=end_day_train.isoformat(),
                                     interval='1d')
        test_df = yfinance.download(tickers=ticker, start=start_day_test.isoformat(), end=end_day_test.isoformat(),
                                    interval='1d')
        if train_df.shape[0] < min_data or test_df.shape[0] < min_data:
            logger.error(f'Cannot get data')
            continue
        model = TurningPointsModelBuilder(ticker=ticker, target_ma_window=target_ema_window, lookahead=look_ahead,
                                          delta=delta)
        boost_model, accuracy = model.train_model(df=train_df)
        bt = Backtest(data=test_df, strategy=MLTurningPointStrategy, commission=commission, exclusive_orders=True,
                      cash=start_cash)
        bt_stats = bt.run(boost_model=boost_model, features_ema_window=features_ema_window)
        logger.info(f'ticker = {ticker},bt_stats : \n {bt_stats}')
        overall_stats = {'ticker': ticker, 'model_accuracy': accuracy, 'SQN': bt_stats['SQN']}
        logger.info(overall_stats)
        logger.info('--------------------------------------------')
