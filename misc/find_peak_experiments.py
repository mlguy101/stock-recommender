# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html

import numpy as np
import matplotlib.pyplot as plt
from scipy.datasets import electrocardiogram
from scipy.signal import find_peaks

if __name__=='__main__':
    x = electrocardiogram()[17000:18000]
    # peaks, _ = find_peaks(x, height=0)
    # plt.plot(x)
    # plt.plot(peaks, x[peaks], "x")
    # plt.plot(np.zeros_like(x), "--", color="gray")
    # plt.show()

    peaks, _ = find_peaks(x, prominence=0.3)
    peaksn,_ = find_peaks(-x,prominence=0.3)
    # np.diff(peaks)
    # array([186, 180, 177, 171, 177, 169, 167, 164, 158, 162, 172])
    plt.plot(x)
    plt.plot(peaks, x[peaks], "x")
    plt.plot(peaksn, x[peaksn], "o")
    plt.show()