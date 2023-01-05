import pandas as pd


def backtest_strategy(df: pd.DataFrame):
    tot_profit = 0
    nrows = df.shape[0]
    last_buy_price = -1
    for i in range(nrows):
        if df.iloc[i]['action'] == 'buy':
            last_buy_price = df.iloc[i]['Adj Close']
        if df.iloc[i]['action'] == 'sell':
            if last_buy_price > 0:
                tot_profit += df.iloc[i]['Adj Close'] - last_buy_price
    return tot_profit
