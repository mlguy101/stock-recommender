import numpy as np
import pandas as pd
from talipp.indicators import EMA
from matplotlib import pyplot as plt
import plotly.express as px

def generate_y(df: pd.DataFrame, ma_window=5, colname='Adj Close'):
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
    # plt.plot(df.index, df['ema'])
    fig = px.line(data_frame=df,x=df.index,y='ema')
    fig.show()
    return df
