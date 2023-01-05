# https://medium.com/geekculture/beating-the-market-with-a-momentum-trading-strategy-using-python-how-you-can-too-18195a9b23fd
import numpy as np
import pandas as pd
import pandas_datareader.data
from datetime import date, timedelta
import yfinance as yf
from matplotlib import pyplot as plt
from backtest import  backtest_strategy
"""
Resources
1- yfinance github https://github.com/ranaroussi/yfinance
2- tutorial https://aroussi.com/post/python-yahoo-finance 
"""


def get_action(num: int):
    if num == 1:
        return 'buy'
    elif num == -1:
        return 'sell'
    elif num == 0:
        return 'hold'
    else:
        return 'unknown'


if __name__ == '__main__':
    # TODO
    """
    Possible api / module parameters 
    1 - analysis window
    2-  short/long window length ( input or recommended based on back testing / ml or opt) 
    3- data period ( hourly , daily ) 
    """
    # data-loader
    stock = 'AAPL'
    start_date = (date.today() - timedelta(days=360)).isoformat()
    end_date = date.today().isoformat()
    print(f'getting data for {stock} from {start_date} to {end_date}')
    data = yf.download(tickers=stock, start=start_date, end=end_date, interval='60m')
    print('got the data')
    small_window = 5
    big_window = 72
    data[f'price_mavg_{small_window}'] = data['Adj Close'].rolling(window=small_window).mean()
    data[f'price_mavg_{big_window}'] = data['Adj Close'].rolling(window=big_window).mean()
    data.dropna(inplace=True)
    plt.xticks(rotation=90)
    plt.plot(data.index, data[f'price_mavg_{small_window}'], label=f'price_mavg_{small_window}')
    plt.plot(data[f'price_mavg_{big_window}'], label=f'price_mavg_{big_window}')
    plt.legend(loc="upper right")
    plt.savefig('fig.png')
    # create signals
    data['pos'] = data[[f'price_mavg_{small_window}', f'price_mavg_{big_window}']] \
        .apply(lambda x: 0 if x[0] < x[1] else 1, axis=1)
    data['pos_diff'] = data['pos'].diff()
    data['action'] = data['pos_diff'].apply(lambda x: get_action(x))
    data.to_csv('data.csv', sep=',', index=True)
    tot_profit = backtest_strategy(df=data)
    print(f'tot profit = {tot_profit}')

