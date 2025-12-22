import numpy as np
from scipy.signal import butter, sosfiltfilt, filtfilt


def design_bandpass_filter(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sos


def design_lowpass_filter(cutoff, fs, order=10):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def apply_filter(signals, sos=None, b=None, a=None, use_sos=True):
    filtered_signals = []

    for session_idx in range(len(signals)):
        sig = signals[session_idx]
        n_trials = sig.shape[0]
        filtered_sig = np.zeros_like(sig)

        for trial_idx in range(n_trials):
            if use_sos:
                filtered_sig[trial_idx, :] = sosfiltfilt(sos, sig[trial_idx, :])
            else:
                filtered_sig[trial_idx, :] = filtfilt(b, a, sig[trial_idx, :])

        filtered_signals.append(filtered_sig)

    return filtered_signals


def downsample(signals, original_fs, target_fs):
    downsample_factor = int(original_fs // target_fs)
    downsampled_signals = []

    for session_idx in range(len(signals)):
        sig = signals[session_idx]
        downsampled_sig = sig[:, ::downsample_factor]
        downsampled_signals.append(downsampled_sig)

    return downsampled_signals


def baseline_correction(signals, fs, baseline_start=0, baseline_end=100):
    baseline_start_idx = int(baseline_start * fs / 1000)
    baseline_end_idx = int(baseline_end * fs / 1000)

    corrected_signals = []

    for session_idx in range(len(signals)):
        sig = signals[session_idx]
        n_trials = sig.shape[0]
        corrected_sig = np.zeros_like(sig)

        for trial_idx in range(n_trials):
            baseline_mean = np.mean(sig[trial_idx, baseline_start_idx:baseline_end_idx])
            corrected_sig[trial_idx, :] = sig[trial_idx, :] - baseline_mean

        corrected_signals.append(corrected_sig)

    return corrected_signals


def preprocess_pipeline(signals, original_fs=10000, target_fs=1000,
                        lowcut=0.1, highcut=500, filter_order=4,
                        baseline_start=0, baseline_end=100):
    # Design and apply bandpass filter
    sos = design_bandpass_filter(lowcut, highcut, original_fs, filter_order)
    filtered = apply_filter(signals, sos=sos, use_sos=True)

    # Downsample
    downsampled = downsample(filtered, original_fs, target_fs)

    # Baseline correction
    corrected = baseline_correction(downsampled, target_fs, baseline_start, baseline_end)

    return corrected
