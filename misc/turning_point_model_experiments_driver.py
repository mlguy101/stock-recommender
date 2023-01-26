import pandas as pd
import yfinance
from datetime import date, timedelta

from backtesting import Backtest, Strategy
from talipp.indicators import EMA


class MLTurningPointStrategy(Strategy):
    bst = None
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

            y_pred = bst.predict(last_record)
            y_pred_round = [round(y_hat) for y_hat in y_pred][0]
            if y_pred_round == TurningPoint.LOCAL_MINIMA.value:
                self.buy()
            elif y_pred_round == TurningPoint.LOCAL_MAXIMA.value:
                self.sell()


from ml.models_builder import TurningPointsModelBuilder
from ml.strategy_builder import HyperParamLocalMinMaxStrategy, TurningPoint

if __name__ == '__main__':
    ## params ####
    features_ema_window = 5
    target_ema_window = 3
    look_ahead=4
    ticker = 'AAPL'
    days = 360
    #####################
    df = yfinance.download(tickers='AAPL', start=(date.today() - timedelta(days)).isoformat(),
                           end=date.today().isoformat(), interval='1d')
    m = TurningPointsModelBuilder(target_ma_window=target_ema_window, lookahead=look_ahead)
    bst = m.train_model(df=df)
    bt = Backtest(data=df, strategy=MLTurningPointStrategy, commission=0.0005, exclusive_orders=True, cash=10000)
    stats = bt.run(bst=bst, features_ema_window=features_ema_window)
    sqn = stats['SQN']
    print(sqn)
