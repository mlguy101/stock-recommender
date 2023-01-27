import datetime
import logging
import pickle
from typing import List
from pathlib import Path
import pandas as pd
import requests
from tqdm import tqdm
from ml.models_builder import TurningModelBuilder, TurningPointsModelBuilder
from datetime import timedelta


def generate_peak_hopt(tickers: List[str], start_datetime: str, end_datetime: str, interval: str,
                       top_n: int | None = None) -> List[dict]:
    ticker_hopt = []
    failure_count = 0
    top_n = min(top_n, len(tickers)) if top_n else None
    tickers = tickers[:top_n] if top_n else tickers

    if top_n:
        logger.info(f'Getting top {top_n} tickers')
    else:
        logger.info(f'Getting all tickers')
    for ticker in tqdm(tickers, desc='iterate over sp500'):
        logger.info(f'Getting peak hopt for ticker = {ticker}')
        ltb = TurningModelBuilder(ticker=ticker)
        ret = ltb.peak_finder_hopt(start=start_datetime, end=end_datetime, interval=interval)
        opt_stats  = ret['opt_stats']
        logger.info(f'Optimal stats for ticker = {ticker} : \n {opt_stats}')
        if ret['opt_stats'] is None:
            failure_count += 1
        ticker_hopt.append(ret)
    return ticker_hopt


if __name__ == '__main__':
    # params #
    ticker_hopt_filename = 'ticker_peak_hopt.pkl'
    start_datetime = (datetime.date.today() - timedelta(150)).isoformat()
    end_datetime = datetime.date.today().isoformat()
    # intervals 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
    # https://algotrading101.com/learn/yfinance-guide/
    interval = '1d'
    ####
    # 1) Get SP500
    # https://gist.github.com/philshem/f2fc94d7e49f045fe0feda8532ab2c08#file-sp-csv
    logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%Y-%m-%d:%H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger()
    logger.info(f'Getting SP00 tickers')
    sp500_df = pd.read_html(requests.get('https://www.slickcharts.com/sp500',
                                         headers={'User-agent': 'Mozilla/5.0'}).text)[0]
    sp500_df.sort_values(by='Weight', inplace=True, ascending=False)

    # load or generate tickers_peak_hopt

    peak_hopts = None
    file_path = Path(f'./{ticker_hopt_filename}')
    if file_path.exists() and file_path.is_file():
        logger.info(f'ticker_hopt already exists at {file_path.name},loading!')
        peak_hopts = pickle.load(file=open(file=file_path, mode='rb'))
    else:
        peak_hopts = generate_peak_hopt(tickers=sp500_df['Symbol'].values, start_datetime=start_datetime,
                                        end_datetime=end_datetime, interval=interval)
        pickle.dump(obj=peak_hopts, file=open(file=ticker_hopt_filename, mode='wb'))

    for entry in peak_hopts:
        strategy = entry['opt_stats']._strategy if entry['opt_stats'] is not None else None
        logger.info(f"""For ticker {entry['ticker']} , optimal peak params = {strategy}""")
