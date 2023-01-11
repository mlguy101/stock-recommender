import yfinance as yf
from datetime import date, timedelta

from ml.y_gen import generate_y

if __name__ == '__main__':
    ## params
    ticker = 'AAPL'
    interval = '60m'
    ##
    start = (date.today() - timedelta(days=180)).isoformat()
    end = date.today().isoformat()
    df = yf.download(tickers=ticker, start=start, end=end,interval=interval)
    df = generate_y(df=df,ma_window=5)

