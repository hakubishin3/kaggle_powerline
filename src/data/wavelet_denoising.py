import pywt
import numpy as np
from scipy.signal import butter, sosfilt


# Signal characteristics
SAMPLING_FREQ = 80000 / 0.02   # 80,000 data points taken over 20 ms


def high_pass_filter(signals, low_freq=10000, sample_fs=SAMPLING_FREQ):
    sos = butter(10, low_freq, btype='highpass', fs=sample_fs, output='sos')
    filtered_sig = sosfilt(sos, signals)

    return filtered_sig


def maddest(d, axis=None):
    """
    Mean Absolute Deviation
    """
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)


def denoise_signal(x, wavelet='db4', level=1):
    coeff = pywt.wavedec(x, wavelet, mode="per")

    for coef_level in range(1, len(coeff)):
        uthresh = (1 / 0.6745) * maddest(np.abs(coeff[-coef_level])) * np.sqrt(2 * np.log(len(x)))
        coeff[-coef_level] = pywt.threshold(coeff[-coef_level], value=uthresh, mode='hard')

    return pywt.waverec(coeff, wavelet, mode='per')
