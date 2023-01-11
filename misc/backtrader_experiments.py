# https://algotrading101.com/learn/backtrader-for-backtesting/
import backtrader as bt
from datetime import date, timedelta
import yfinance as yf


class PrintClose(bt.Strategy):

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
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
    df.to_csv('./data.csv', sep=',')
    data = bt.feeds.YahooFinanceCSVData(dataname='data.csv')

    cerebro.adddata(data)
    # Run Cerebro Engine
    cerebro.run()
