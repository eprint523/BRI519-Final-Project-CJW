"""
Outlier Detection Module

Functions for detecting and removing outlier trials based on signal variance.
"""

import numpy as np


def detect_outliers_mad(data, k=3):
    """
    Detect outliers using Median Absolute Deviation (MAD) method.

    The MAD is a robust measure of variability. Outliers are defined as
    data points that fall outside the range [median - k*MAD, median + k*MAD].

    Parameters
    ----------
    data : numpy.ndarray
        1D array of values (e.g., trial variances).
    k : float, optional
        Number of MADs from median to define outlier threshold (default: 3).

    Returns
    -------
    outlier_mask : numpy.ndarray
        Boolean array where True indicates an outlier.
    median : float
        Median of the data.
    mad : float
        Median Absolute Deviation.
    lower_bound : float
        Lower threshold for outlier detection.
    upper_bound : float
        Upper threshold for outlier detection.
    """
    median = np.median(data)
    mad = np.median(np.abs(data - median))

    lower_bound = median - k * mad
    upper_bound = median + k * mad

    outlier_mask = (data < lower_bound) | (data > upper_bound)

    return outlier_mask, median, mad, lower_bound, upper_bound


def compute_trial_variances(signals):
    """
    Compute variance for each trial across all sessions.

    Parameters
    ----------
    signals : list
        List of signal arrays (one per session).

    Returns
    -------
    variances : list
        List of variance arrays (one per session).
    """
    variances = []
    for session_idx in range(len(signals)):
        sig = signals[session_idx]
        var = np.var(sig, axis=1)
        variances.append(var)
    return variances


def find_outlier_indices(signals, k=3):
    """
    Find outlier trial indices for each session.

    Parameters
    ----------
    signals : list
        List of signal arrays (one per session).
    k : float, optional
        MAD multiplier for outlier threshold (default: 3).

    Returns
    -------
    outlier_indices : list
        List of arrays containing outlier trial indices for each session.
    detection_info : list
        List of dictionaries with detection statistics for each session.
    """
    outlier_indices = []
    detection_info = []

    for session_idx in range(len(signals)):
        sig = signals[session_idx]
        variances = np.var(sig, axis=1)

        outlier_mask, median, mad, lower_bound, upper_bound = detect_outliers_mad(variances, k=k)
        outlier_idx = np.where(outlier_mask)[0]

        outlier_indices.append(outlier_idx)
        detection_info.append({
            'session': session_idx,
            'n_trials': sig.shape[0],
            'n_outliers': len(outlier_idx),
            'median': median,
            'mad': mad,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'variances': variances,
            'outlier_mask': outlier_mask
        })

    return outlier_indices, detection_info


def remove_outliers(signals, outlier_indices):
    """
    Remove outlier trials from signals.

    Parameters
    ----------
    signals : list
        List of signal arrays (one per session).
    outlier_indices : list
        List of arrays containing outlier trial indices for each session.

    Returns
    -------
    clean_signals : list
        List of signal arrays with outliers removed.
    removal_summary : list
        List of dictionaries with removal statistics for each session.
    """
    clean_signals = []
    removal_summary = []

    for session_idx in range(len(signals)):
        sig = signals[session_idx]
        outlier_idx = outlier_indices[session_idx]

        # Create mask for trials to keep
        keep_mask = np.ones(sig.shape[0], dtype=bool)
        keep_mask[outlier_idx] = False

        clean_sig = sig[keep_mask, :]
        clean_signals.append(clean_sig)

        removal_summary.append({
            'session': session_idx,
            'before': sig.shape[0],
            'after': clean_sig.shape[0],
            'removed': len(outlier_idx)
        })

    return clean_signals, removal_summary


def print_removal_summary(removal_summary_low, removal_summary_high):
    """
    Print a formatted summary of outlier removal.

    Parameters
    ----------
    removal_summary_low : list
        Removal summary for low frequency condition.
    removal_summary_high : list
        Removal summary for high frequency condition.
    """
    print("Outlier Removal Summary")
    print("=" * 60)

    print("\nLow Frequency:")
    total_before_low = 0
    total_after_low = 0
    for summary in removal_summary_low:
        print(f"  Session {summary['session']}: {summary['before']} trials -> "
              f"{summary['after']} trials (removed {summary['removed']})")
        total_before_low += summary['before']
        total_after_low += summary['after']

    print("\nHigh Frequency:")
    total_before_high = 0
    total_after_high = 0
    for summary in removal_summary_high:
        print(f"  Session {summary['session']}: {summary['before']} trials -> "
              f"{summary['after']} trials (removed {summary['removed']})")
        total_before_high += summary['before']
        total_after_high += summary['after']

    print("\nTotal:")
    removed_low = total_before_low - total_after_low
    removed_high = total_before_high - total_after_high
    print(f"  Low frequency: {total_before_low} -> {total_after_low} "
          f"(removed {removed_low}, {removed_low/total_before_low*100:.1f}%)")
    print(f"  High frequency: {total_before_high} -> {total_after_high} "
          f"(removed {removed_high}, {removed_high/total_before_high*100:.1f}%)")
