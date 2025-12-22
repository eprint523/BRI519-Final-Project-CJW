import numpy as np
from scipy import signal as sp_signal
from scipy import stats


# Default frequency bands for LFP analysis
DEFAULT_FREQ_BANDS = {
    'Delta': (1, 4),
    'Theta': (4, 8),
    'Alpha': (8, 12),
    'Beta': (13, 30),
    'Low Gamma': (30, 80),
    'High Gamma': (80, 200)
}


def compute_erp(signals, fs=1000):
    erp_results = []

    for session_idx in range(len(signals)):
        sig = signals[session_idx]
        n_trials = sig.shape[0]
        n_samples = sig.shape[1]

        mean_erp = np.mean(sig, axis=0)
        sem_erp = np.std(sig, axis=0) / np.sqrt(n_trials)
        time_ms = np.arange(n_samples) / fs * 1000

        erp_results.append({
            'session': session_idx,
            'mean': mean_erp,
            'sem': sem_erp,
            'time_ms': time_ms,
            'n_trials': n_trials
        })

    return erp_results


def find_erp_peak(erp_mean, fs=1000, window_start=100, window_end=200, peak_type='min'):
    start_idx = int(window_start * fs / 1000)
    end_idx = int(window_end * fs / 1000)

    if peak_type == 'min':
        peak_idx = np.argmin(erp_mean[start_idx:end_idx]) + start_idx
    else:
        peak_idx = np.argmax(erp_mean[start_idx:end_idx]) + start_idx

    peak_amplitude = erp_mean[peak_idx]
    peak_latency = peak_idx / fs * 1000

    return peak_amplitude, peak_latency


def compute_psd(signals, fs=1000, nperseg=128):
    psd_results = []

    for session_idx in range(len(signals)):
        sig = signals[session_idx]
        n_trials = sig.shape[0]

        # Compute PSD for each trial and average
        psd_all = []
        for trial_idx in range(n_trials):
            freqs, psd = sp_signal.welch(sig[trial_idx, :], fs=fs, nperseg=nperseg)
            psd_all.append(psd)

        mean_psd = np.mean(psd_all, axis=0)
        sem_psd = np.std(psd_all, axis=0) / np.sqrt(n_trials)

        psd_results.append({
            'session': session_idx,
            'freqs': freqs,
            'mean_psd': mean_psd,
            'sem_psd': sem_psd,
            'n_trials': n_trials
        })

    return psd_results


def compute_peak_frequency(psd_result, freq_range=(10, 200)):
    freqs = psd_result['freqs']
    mean_psd = psd_result['mean_psd']

    freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    masked_freqs = freqs[freq_mask]
    masked_psd = mean_psd[freq_mask]

    peak_idx = np.argmax(masked_psd)
    peak_freq = masked_freqs[peak_idx]
    peak_power = masked_psd[peak_idx]

    return peak_freq, peak_power


def compute_band_power(signals, fs=1000, freq_bands=None, nperseg=128):
    if freq_bands is None:
        freq_bands = DEFAULT_FREQ_BANDS

    band_power_results = []

    for session_idx in range(len(signals)):
        sig = signals[session_idx]
        n_trials = sig.shape[0]

        # Initialize storage for each band
        band_powers = {band: np.zeros(n_trials) for band in freq_bands.keys()}

        for trial_idx in range(n_trials):
            freqs, psd = sp_signal.welch(sig[trial_idx, :], fs=fs, nperseg=nperseg)

            for band_name, (low_freq, high_freq) in freq_bands.items():
                band_idx = np.where((freqs >= low_freq) & (freqs <= high_freq))[0]

                if len(band_idx) > 0:
                    # Integrate power in band using trapezoidal rule
                    band_power = np.trapezoid(psd[band_idx], freqs[band_idx])
                    band_powers[band_name][trial_idx] = band_power

        band_power_results.append({
            'session': session_idx,
            'band_powers': band_powers,
            'n_trials': n_trials
        })

    return band_power_results


def compute_band_ratio(band_power_result, numerator_band, denominator_band):
    band_powers = band_power_result['band_powers']
    numerator = band_powers[numerator_band]
    denominator = band_powers[denominator_band]

    # Add small epsilon to avoid division by zero
    ratio = numerator / (denominator + 1e-10)

    return ratio


def compare_conditions(data_low, data_high, test='ttest'):
    if test == 'ttest':
        statistic, p_value = stats.ttest_ind(data_low, data_high)
    elif test == 'mannwhitney':
        statistic, p_value = stats.mannwhitneyu(data_low, data_high)
    else:
        raise ValueError(f"Unknown test: {test}")

    return statistic, p_value


def analyze_band_power_statistics(band_power_low, band_power_high, freq_bands=None):
    if freq_bands is None:
        freq_bands = DEFAULT_FREQ_BANDS

    stats_results = []

    for session_idx in range(len(band_power_low)):
        powers_low = band_power_low[session_idx]['band_powers']
        powers_high = band_power_high[session_idx]['band_powers']

        session_stats = {'session': session_idx, 'bands': {}}

        for band_name in freq_bands.keys():
            mean_low = np.mean(powers_low[band_name])
            mean_high = np.mean(powers_high[band_name])
            t_stat, p_val = stats.ttest_ind(powers_low[band_name], powers_high[band_name])

            session_stats['bands'][band_name] = {
                'mean_low': mean_low,
                'mean_high': mean_high,
                'ratio': mean_low / (mean_high + 1e-10),
                't_statistic': t_stat,
                'p_value': p_val,
                'significant': p_val < 0.05
            }

        stats_results.append(session_stats)

    return stats_results
