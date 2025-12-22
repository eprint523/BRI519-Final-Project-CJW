import numpy as np
import scipy.io


def load_lfp_data(filepath):
    data = scipy.io.loadmat(filepath)
    raw_sig = data['DATA'][:, 0]
    raw_freq = data['DATA'][:, 4]

    return raw_sig, raw_freq


def separate_by_frequency(raw_sig, raw_freq, n_sessions=4):
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
    return [(f"{low:.0f} Hz", f"{high:.0f} Hz") for low, high in freq_info]
