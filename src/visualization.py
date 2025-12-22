"""
Visualization Module

Functions for creating plots and figures for LFP analysis.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_variance_distribution(detection_info, title_suffix="", save_path=None):
    """
    Plot variance distribution and outlier detection results.

    Parameters
    ----------
    detection_info : list
        List of detection info dictionaries from find_outlier_indices.
    title_suffix : str, optional
        Suffix to add to plot title.
    save_path : str, optional
        Path to save figure. If None, figure is displayed.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Variance Distribution and Outlier Detection {title_suffix}', fontsize=16)

    for info in detection_info:
        session_idx = info['session']
        ax = axes[session_idx // 2, session_idx % 2]

        n_trials = info['n_trials']
        variances = info['variances']
        outlier_mask = info['outlier_mask']

        trial_indices = np.arange(n_trials)
        colors = ['red' if outlier else 'blue' for outlier in outlier_mask]

        ax.scatter(trial_indices, variances, c=colors, alpha=0.6, s=30)
        ax.axhline(info['median'], color='green', linestyle='--', linewidth=2,
                   label=f"Median: {info['median']:.2f}")
        ax.axhline(info['upper_bound'], color='orange', linestyle='--', linewidth=1.5,
                   label=f"Upper: {info['upper_bound']:.2f}")
        ax.axhline(info['lower_bound'], color='orange', linestyle='--', linewidth=1.5,
                   label=f"Lower: {info['lower_bound']:.2f}")

        ax.set_xlabel('Trial Index')
        ax.set_ylabel('Variance')
        ax.set_title(f"Session {session_idx} (n={n_trials}, outliers={info['n_outliers']})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_outlier_comparison(signals_before, outlier_indices, title_suffix="", save_path=None):
    """
    Plot comparison of normal vs outlier trial waveforms.

    Parameters
    ----------
    signals_before : list
        List of signal arrays before outlier removal.
    outlier_indices : list
        List of outlier index arrays.
    title_suffix : str, optional
        Suffix to add to plot title.
    save_path : str, optional
        Path to save figure.
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f'Normal vs Outlier Trial Examples {title_suffix}', fontsize=14)

    n_samples = signals_before[0].shape[1]
    time_ms = np.arange(n_samples)

    for session_idx in range(4):
        sig = signals_before[session_idx]
        outlier_idx = outlier_indices[session_idx]
        normal_idx = np.setdiff1d(np.arange(sig.shape[0]), outlier_idx)

        # Normal trials (top row)
        ax_normal = axes[0, session_idx]
        for i in normal_idx[:5]:
            ax_normal.plot(time_ms, sig[i, :], alpha=0.6, linewidth=0.8)
        ax_normal.axvline(100, color='green', linestyle='--', linewidth=1, alpha=0.7)
        ax_normal.set_title(f'Session {session_idx} - Normal (n={len(normal_idx)})')
        ax_normal.set_xlabel('Time (ms)')
        ax_normal.set_ylabel('Amplitude')
        ax_normal.grid(True, alpha=0.3)

        # Outlier trials (bottom row)
        ax_outlier = axes[1, session_idx]
        for i in outlier_idx[:5]:
            ax_outlier.plot(time_ms, sig[i, :], alpha=0.6, linewidth=0.8, color='red')
        ax_outlier.axvline(100, color='green', linestyle='--', linewidth=1, alpha=0.7)
        ax_outlier.set_title(f'Session {session_idx} - Outlier (n={len(outlier_idx)})')
        ax_outlier.set_xlabel('Time (ms)')
        ax_outlier.set_ylabel('Amplitude')
        ax_outlier.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_before_after_removal(signals_before, signals_after, title_suffix="", save_path=None):
    """
    Plot mean signal before and after outlier removal.

    Parameters
    ----------
    signals_before : list
        List of signal arrays before outlier removal.
    signals_after : list
        List of signal arrays after outlier removal.
    title_suffix : str, optional
        Suffix to add to plot title.
    save_path : str, optional
        Path to save figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Mean Signal: Before vs After Outlier Removal {title_suffix}', fontsize=14)

    n_samples = signals_before[0].shape[1]
    time_ms = np.arange(n_samples)

    for session_idx in range(4):
        ax = axes[session_idx // 2, session_idx % 2]

        sig_before = signals_before[session_idx]
        sig_after = signals_after[session_idx]

        mean_before = np.mean(sig_before, axis=0)
        mean_after = np.mean(sig_after, axis=0)

        ax.plot(time_ms, mean_before, 'r-', linewidth=2,
                label=f'Before (n={sig_before.shape[0]})', alpha=0.8)
        ax.plot(time_ms, mean_after, 'b-', linewidth=2,
                label=f'After (n={sig_after.shape[0]})', alpha=0.8)
        ax.axvline(100, color='green', linestyle='--', linewidth=1.5, label='Stim Onset')
        ax.axvline(150, color='orange', linestyle='--', linewidth=1.5, label='Stim Offset')
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'Session {session_idx}')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_erp(erp_low, erp_high, freq_info, stim_onset=100, stim_offset=150, save_path=None):
    """
    Plot ERP comparison between low and high frequency conditions.

    Parameters
    ----------
    erp_low : list
        ERP results for low frequency condition.
    erp_high : list
        ERP results for high frequency condition.
    freq_info : list
        Frequency information for each session.
    stim_onset : float, optional
        Stimulus onset time in ms (default: 100).
    stim_offset : float, optional
        Stimulus offset time in ms (default: 150).
    save_path : str, optional
        Path to save figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for session_idx in range(4):
        ax = axes[session_idx // 2, session_idx % 2]

        erp_l = erp_low[session_idx]
        erp_h = erp_high[session_idx]
        time_ms = erp_l['time_ms']

        # Plot mean with SEM
        ax.plot(time_ms, erp_l['mean'], color='darkblue', linewidth=3,
                label=f"Low Freq (n={erp_l['n_trials']})")
        ax.fill_between(time_ms, erp_l['mean'] - erp_l['sem'],
                        erp_l['mean'] + erp_l['sem'], color='blue', alpha=0.25)

        ax.plot(time_ms, erp_h['mean'], color='darkred', linewidth=3,
                label=f"High Freq (n={erp_h['n_trials']})")
        ax.fill_between(time_ms, erp_h['mean'] - erp_h['sem'],
                        erp_h['mean'] + erp_h['sem'], color='red', alpha=0.25)

        # Event markers
        ax.axvline(stim_onset, color='green', linestyle='--', linewidth=2, label='Stim Onset')
        ax.axvline(stim_offset, color='orange', linestyle='--', linewidth=1.5, label='Stim Offset')
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

        freq_diff = freq_info[session_idx][1] - freq_info[session_idx][0]
        ax.set_xlabel('Time (ms)', fontsize=11)
        ax.set_ylabel('Amplitude', fontsize=11)
        ax.set_title(f'Session {session_idx}: {freq_info[session_idx][0]:.0f} Hz vs '
                     f'{freq_info[session_idx][1]:.0f} Hz\nFrequency Difference: {freq_diff:.0f} Hz')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 300])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_psd(psd_low, psd_high, freq_info, save_path=None):
    """
    Plot Power Spectral Density comparison.

    Parameters
    ----------
    psd_low : list
        PSD results for low frequency condition.
    psd_high : list
        PSD results for high frequency condition.
    freq_info : list
        Frequency information for each session.
    save_path : str, optional
        Path to save figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Power Spectral Density Analysis', fontsize=14)

    for session_idx in range(4):
        ax = axes[session_idx // 2, session_idx % 2]

        psd_l = psd_low[session_idx]
        psd_h = psd_high[session_idx]

        ax.semilogy(psd_l['freqs'], psd_l['mean_psd'], 'b-', linewidth=2,
                    label=f"Low Freq ({freq_info[session_idx][0]:.0f} Hz)")
        ax.semilogy(psd_h['freqs'], psd_h['mean_psd'], 'r-', linewidth=2,
                    label=f"High Freq ({freq_info[session_idx][1]:.0f} Hz)")

        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power Spectral Density')
        ax.set_title(f'Session {session_idx}')
        ax.set_xlim([0, 250])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_band_power(band_power_low, band_power_high, freq_info, freq_bands, save_path=None):
    """
    Plot band power comparison as bar charts.

    Parameters
    ----------
    band_power_low : list
        Band power results for low frequency condition.
    band_power_high : list
        Band power results for high frequency condition.
    freq_info : list
        Frequency information for each session.
    freq_bands : dict
        Dictionary of frequency bands.
    save_path : str, optional
        Path to save figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    band_names = list(freq_bands.keys())
    x_pos = np.arange(len(band_names))
    width = 0.35

    for session_idx in range(4):
        ax = axes[session_idx // 2, session_idx % 2]

        powers_low = band_power_low[session_idx]['band_powers']
        powers_high = band_power_high[session_idx]['band_powers']

        mean_low = [np.mean(powers_low[band]) for band in band_names]
        sem_low = [np.std(powers_low[band]) / np.sqrt(len(powers_low[band])) for band in band_names]

        mean_high = [np.mean(powers_high[band]) for band in band_names]
        sem_high = [np.std(powers_high[band]) / np.sqrt(len(powers_high[band])) for band in band_names]

        ax.bar(x_pos - width/2, mean_low, width, yerr=sem_low, capsize=5, alpha=0.8,
               label=f"Low Freq ({freq_info[session_idx][0]:.0f} Hz)", color='blue')
        ax.bar(x_pos + width/2, mean_high, width, yerr=sem_high, capsize=5, alpha=0.8,
               label=f"High Freq ({freq_info[session_idx][1]:.0f} Hz)", color='red')

        ax.set_xlabel('Frequency Band', fontsize=11)
        ax.set_ylabel('Power', fontsize=11)
        ax.set_title(f'Session {session_idx}')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(band_names, rotation=45, ha='right')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_yscale('log')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_band_ratio(ratio_low, ratio_high, ratio_name, save_path=None):
    """
    Plot band power ratio comparison.

    Parameters
    ----------
    ratio_low : list
        Ratio values for low frequency condition (one per session).
    ratio_high : list
        Ratio values for high frequency condition (one per session).
    ratio_name : str
        Name of the ratio being plotted.
    save_path : str, optional
        Path to save figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    sessions = np.arange(4)
    width = 0.35

    ax.bar(sessions - width/2, ratio_low, width, label='Low Tone', color='blue', alpha=0.8)
    ax.bar(sessions + width/2, ratio_high, width, label='High Tone', color='red', alpha=0.8)

    ax.set_xlabel('Session')
    ax.set_ylabel('Power Ratio')
    ax.set_title(f'{ratio_name} Ratio')
    ax.set_xticks(sessions)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
