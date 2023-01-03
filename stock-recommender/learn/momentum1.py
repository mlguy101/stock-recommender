# https://medium.com/codex/picking-stocks-with-a-quantitative-momentum-strategy-in-python-b15ac8925ec6

import pandas as pd
import numpy as np
from scipy.stats import percentileofscore as score
from intraday_data import get_intraday_prices
from tqdm import tqdm

if __name__ == '__main__':
    sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    sp500_list = np.array(sp500[0]['Symbol'])
    n = 5
    print(sp500_list[:n])

    df = pd.DataFrame(columns=sp500_list[:n])
    for i in tqdm(df.columns):
        print(f'getting data for {i}')
        try:
            df[i] = get_intraday_prices(i)['Close']
            print(f'{i} is successfully extracted')
        except:
            raise Exception(f"{i}")

    dc = []
    for i in df.columns:
        dc.append(df[i].pct_change().sum())

    sp500_momentum = pd.DataFrame(columns=['symbol', 'day_change'])
    sp500_momentum['symbol'] = df.columns
    sp500_momentum['day_change'] = dc

    sp500_momentum['momentum'] = 'N/A'
    for i in range(len(sp500_momentum)):
        sp500_momentum.loc[i, 'momentum'] = score(sp500_momentum.day_change, sp500_momentum.loc[i, 'day_change']) / 100

    sp500_momentum['momentum'] = sp500_momentum['momentum'].astype(float)
    sp500_momentum.sort_values(by='momentum',ascending=False,inplace=True)
    print(sp500_momentum)
