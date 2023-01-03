import requests
import pandas as pd

def get_intraday_prices(symbol):
    ticker = symbol
    iex_api_key = 'sk_733cd495b4f3457baee3cbd45e7cf148'
    url = f'https://cloud.iexapis.com/stable/stock/{ticker}/intraday-prices?token={iex_api_key}'
    df = requests.get(url).json()
    date = df[1]['date']
        
    time = []
    open = []
    high = []
    low = []
    close = []
    volume = []
    number_of_trades = []
    
    for i in range(len(df)):
        time.append(df[i]['label'])
        open.append(df[i]['open'])
        high.append(df[i]['high'])
        low.append(df[i]['low'])
        close.append(df[i]['close'])
        volume.append(df[i]['volume'])
        number_of_trades.append(df[i]['numberOfTrades'])
        
    time_df = pd.DataFrame(time).rename(columns = {0:'Time'})
    open_df = pd.DataFrame(open).rename(columns = {0:'Open'})
    high_df = pd.DataFrame(high).rename(columns = {0:'High'})
    low_df = pd.DataFrame(low).rename(columns = {0:'Low'})
    close_df = pd.DataFrame(close).rename(columns = {0:'Close'})
    volume_df = pd.DataFrame(volume).rename(columns = {0:'Volume'})
    number_of_trades_df = pd.DataFrame(number_of_trades).rename(columns = {0:'Number of Trades'})
     
    frames = [time_df, open_df, high_df, low_df, close_df, volume_df, number_of_trades_df]
    df = pd.concat(frames, axis = 1, join = 'inner')
    df = df.set_index('Time')
    return df
