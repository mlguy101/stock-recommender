import yfinance
from datetime import date, timedelta

from backtesting import Backtest

from ml.models_builder import LocalTurningPointsModelBuilder
from ml.strategy_builder import HyperParamLocalMinMaxStrategy

if __name__ == '__main__':
    # df = yfinance.download(tickers='AAPL', start=(date.today() - timedelta(180)).isoformat(),
    #                        end=date.today().isoformat(), interval='1d')
    # bt = Backtest(data=df, strategy=HyperParamLocalMinMaxStrategy, cash=10000, commission=0.0001,exclusive_orders=True)
    # stats = bt.run()
    # print(stats)
    m = LocalTurningPointsModelBuilder(ticker='AAPL', target_ma_window=3, lookahead=4)
    m.train_model(start_datetime=(date.today() - timedelta(150)).isoformat(), end_datetime=(date.today()),
                  interval='1d')
