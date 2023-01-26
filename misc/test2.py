import yfinance
from datetime import date, timedelta

from backtesting import Backtest

from ml.strategy_builder import HyperParamLocalMinMaxStrategy

if __name__ == '__main__':
    df = yfinance.download(tickers='AAPL', start=(date.today() - timedelta(180)).isoformat(),
                           end=date.today().isoformat(), interval='1d')
    bt = Backtest(data=df, strategy=HyperParamLocalMinMaxStrategy, cash=10000, commission=0.0001,exclusive_orders=True)
    stats = bt.run()
    print(stats)
