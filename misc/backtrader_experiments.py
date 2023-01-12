# https://algotrading101.com/learn/backtrader-for-backtesting/
# https://community.backtrader.com/topic/3979/help-with-yahoofinance-data/2
# https://www.backtrader.com/docu/quickstart/quickstart/
import backtrader as bt
from datetime import date, timedelta
import yfinance as yf


class CustomDataFeed(bt.feeds.PandasData):
    params = (
        ('datetime', None),
        ('open', 'Open'),
        ('high', 'High'),
        ('low', 'Low'),
        ('close', 'Close'),
        ('volume', 'Volume'),

    )


class PrintClose(bt.Strategy):

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] data-series
        self.dataclose = self.datas[0].close

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {txt}')  # Print date and close

    def next(self):
        self.log('Close: ', self.dataclose[0])


if __name__ == '__main__':
    # Instantiate Cerebro engine
    cerebro = bt.Cerebro()

    # Add strategy to Cerebro
    cerebro.addstrategy(PrintClose)
    # add data
    ## params
    ticker = 'AAPL'
    interval = '60m'
    ##
    start = (date.today() - timedelta(days=180)).isoformat()
    end = date.today().isoformat()
    df = yf.download(tickers=ticker, start=start, end=end, interval=interval)

    data = CustomDataFeed(dataname = df)
    cerebro.adddata(data)
    # # Run Cerebro Engine
    cerebro.run()