# https://medium.com/geekculture/beating-the-market-with-a-momentum-trading-strategy-using-python-how-you-can-too-18195a9b23fd
import pandas as pd
import pandas_datareader.data
from datetime import date,timedelta
import yfinance as yf
"""
Resources
1- yfinance github https://github.com/ranaroussi/yfinance
2- tutorial https://aroussi.com/post/python-yahoo-finance 
"""
if __name__ == '__main__':
    # TODO
    """
    Possible api / module parameters 
    1 - analysis window
    2- 
    """
    # data-loader
    stock = 'AAPL'
    start_date = (date.today()-timedelta(days=150)).isoformat()
    end_date = date.today().isoformat()
    data = yf.download(tickers=stock, start=start_date, end=end_date)
    print(data.columns)
    print(data)


