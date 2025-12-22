"""
Data Loader Module

Functions for loading and organizing LFP data from .mat files.
"""

import numpy as np
import scipy.io


def load_lfp_data(filepath):
    """
    Load LFP data from a .mat file.

    Parameters
    ----------
    filepath : str
        Path to the .mat file containing LFP data.

    Returns
    -------
    raw_sig : numpy.ndarray
        Raw signal data array (sessions x trials x samples).
    raw_freq : numpy.ndarray
        Frequency labels for each trial.
    """
    data = scipy.io.loadmat(filepath)
    raw_sig = data['DATA'][:, 0]
    raw_freq = data['DATA'][:, 4]

    return raw_sig, raw_freq


def separate_by_frequency(raw_sig, raw_freq, n_sessions=4):
    """
    Separate signals by stimulus frequency (low vs high tone).

    Parameters
    ----------
    raw_sig : numpy.ndarray
        Raw signal data from each session.
    raw_freq : numpy.ndarray
        Frequency labels for each trial.
    n_sessions : int, optional
        Number of recording sessions (default: 4).

    Returns
    -------
    sig_low : list
        List of signal arrays for low frequency tone trials.
    sig_high : list
        List of signal arrays for high frequency tone trials.
    freq_info : list
        List of tuples containing (low_freq, high_freq) for each session.
    """
    sig_low = []
    sig_high = []
    freq_info = []

    for session_idx in range(n_sessions):
        sig = raw_sig[session_idx]
        freq = raw_freq[session_idx]

        # Identify unique frequencies in this session
        unique_freqs = np.sort(np.unique(freq))
        low_freq = unique_freqs[0]
        high_freq = unique_freqs[1]

        # Separate trials by frequency
        low_freq_idx = np.where(freq == low_freq)[0]
        high_freq_idx = np.where(freq == high_freq)[0]

        sig_low.append(sig[low_freq_idx, :])
        sig_high.append(sig[high_freq_idx, :])
        freq_info.append((low_freq, high_freq))

    return sig_low, sig_high, freq_info


def get_frequency_info_string(freq_info):
    """
    Convert frequency info to string format for display.

    Parameters
    ----------
    freq_info : list
        List of (low_freq, high_freq) tuples.

    Returns
    -------
    list
        List of tuples with frequency strings (e.g., ("9500 Hz", "19000 Hz")).
    """
    return [(f"{low:.0f} Hz", f"{high:.0f} Hz") for low, high in freq_info]
