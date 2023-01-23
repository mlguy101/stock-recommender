"""
1- Get Hopt calibration data
2- calibrate find-peak parameters based on backtestpy optimize
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

from backtesting import Backtest

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
        <ticker>  No timezone found, symbol may be delisted 
        https://github.com/ranaroussi/yfinance/issues/359 
        """
        df = self.get_data(ticker=self.ticker, start=start, end=end, interval=interval)
        if df.shape[0] < 1:
            self.logger.error(f'Cannot download data for ticker {self.ticker}')
            return {'ticker': self.ticker, 'opt_stats': None}
        bt = Backtest(data=df, strategy=HyperParamLocalMinMaxStrategy, commission=0.005, cash=10_000)
        opt_stats = bt.optimize(prominence=[0.4, 0.8], distance=[20, 60, 100, 150],
                                width=[10, 20], ma_window=[5, 10, 15, 20], maximize='Equity Final [$]')

        return {'ticker': self.ticker, 'opt_stats': opt_stats}
