import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from talipp.indicators import EMA
import plotly.express as px
from scipy.signal import find_peaks


def generate_y(df: pd.DataFrame, ma_window=5, colname='Adj Close', prominence=1.0, distance=5):
    """
    :param df:
    :param ma_window:
    :param colname:
    :return:
    """
    # TODO
    """
    https://eddwardo.github.io/posts/2019-06-05-finding-local-extreams-in-pandas-time-series/
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks_cwt.html#scipy.signal.find_peaks_cwt 
    """

    df['ema'] = [np.nan] * (ma_window - 1) + EMA(period=ma_window, input_values=df[colname]).output_values

    distance = 10
    local_maxima_idx, _ = find_peaks(x=df['ema'].values, prominence=prominence, distance=distance)
    local_minima_idx, _ = find_peaks(x=-df['ema'].values, prominence=prominence, distance=distance)
    # plt.plot(df.index, df['ema'])
    # fig = px.line(data_frame=df, x=df.index, y='ema')
    # fig.show()
    plt.plot(df['ema'].values)
    plt.plot(local_maxima_idx, df['ema'].values[local_maxima_idx], 'x')
    plt.plot(local_minima_idx, df['ema'].values[local_minima_idx], 'o')
    plt.show()
    return df
