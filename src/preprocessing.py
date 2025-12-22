"""
Preprocessing Module

Functions for signal preprocessing including filtering, downsampling,
and baseline correction.
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt, filtfilt


def design_bandpass_filter(lowcut, highcut, fs, order=4):
    """
    Design a Butterworth bandpass filter.

    Parameters
    ----------
    lowcut : float
        Low cutoff frequency in Hz.
    highcut : float
        High cutoff frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    order : int, optional
        Filter order (default: 4).

    Returns
    -------
    sos : ndarray
        Second-order sections representation of the filter.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sos


def design_lowpass_filter(cutoff, fs, order=10):
    """
    Design a Butterworth lowpass filter.

    Parameters
    ----------
    cutoff : float
        Cutoff frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    order : int, optional
        Filter order (default: 10).

    Returns
    -------
    b, a : ndarray
        Numerator and denominator of the filter.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def apply_filter(signals, sos=None, b=None, a=None, use_sos=True):
    """
    Apply filter to signals across all sessions and trials.

    Parameters
    ----------
    signals : list
        List of signal arrays (one per session).
    sos : ndarray, optional
        Second-order sections filter coefficients.
    b, a : ndarray, optional
        Filter coefficients for transfer function form.
    use_sos : bool, optional
        If True, use sos format; otherwise use b, a format.

    Returns
    -------
    filtered_signals : list
        List of filtered signal arrays.
    """
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
    """
    Downsample signals to a target sampling frequency.

    Parameters
    ----------
    signals : list
        List of signal arrays (one per session).
    original_fs : float
        Original sampling frequency in Hz.
    target_fs : float
        Target sampling frequency in Hz.

    Returns
    -------
    downsampled_signals : list
        List of downsampled signal arrays.
    """
    downsample_factor = int(original_fs // target_fs)
    downsampled_signals = []

    for session_idx in range(len(signals)):
        sig = signals[session_idx]
        downsampled_sig = sig[:, ::downsample_factor]
        downsampled_signals.append(downsampled_sig)

    return downsampled_signals


def baseline_correction(signals, fs, baseline_start=0, baseline_end=100):
    """
    Apply baseline correction by subtracting mean of baseline period.

    Parameters
    ----------
    signals : list
        List of signal arrays (one per session).
    fs : float
        Sampling frequency in Hz.
    baseline_start : float, optional
        Start of baseline period in ms (default: 0).
    baseline_end : float, optional
        End of baseline period in ms (default: 100).

    Returns
    -------
    corrected_signals : list
        List of baseline-corrected signal arrays.
    """
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
    """
    Run full preprocessing pipeline: filter -> downsample -> baseline correction.

    Parameters
    ----------
    signals : list
        List of raw signal arrays.
    original_fs : float
        Original sampling frequency (default: 10000 Hz).
    target_fs : float
        Target sampling frequency after downsampling (default: 1000 Hz).
    lowcut : float
        Low cutoff for bandpass filter (default: 0.1 Hz).
    highcut : float
        High cutoff for bandpass filter (default: 500 Hz).
    filter_order : int
        Order of the Butterworth filter (default: 4).
    baseline_start : float
        Start of baseline period in ms (default: 0).
    baseline_end : float
        End of baseline period in ms (default: 100).

    Returns
    -------
    processed_signals : list
        List of fully preprocessed signal arrays.
    """
    # Design and apply bandpass filter
    sos = design_bandpass_filter(lowcut, highcut, original_fs, filter_order)
    filtered = apply_filter(signals, sos=sos, use_sos=True)

    # Downsample
    downsampled = downsample(filtered, original_fs, target_fs)

    # Baseline correction
    corrected = baseline_correction(downsampled, target_fs, baseline_start, baseline_end)

    return corrected
