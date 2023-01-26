# https://stackoverflow.com/a/64588944/5937273
# https://stackoverflow.com/questions/1713335/peak-finding-algorithm-for-python-scipy
# https://github.com/avhn/peakdetect
import numpy as np
from matplotlib import pyplot as plt
from peakdetect import peakdetect
import yfinance as yf
from datetime import date, timedelta

from talipp.indicators import EMA

df = yf.download(tickers='AAPL', start=(date.today() - timedelta(150)).isoformat(), end=date.today().isoformat(),
                 interval='1d')

ema = EMA(period=5, input_values=df['Adj Close'].values)
peaks = peakdetect(ema, lookahead=5)
# Lookahead is the distance to look ahead from a peak to determine if it is the actual peak.
# Change lookahead as necessary
higherPeaks = np.array(peaks[0])
lowerPeaks = np.array(peaks[1])
plt.plot(ema)
plt.plot(higherPeaks[:, 0], higherPeaks[:, 1], 'ro')
plt.plot(lowerPeaks[:, 0], lowerPeaks[:, 1], 'ko')
plt.savefig('plot.png')
