import os
import numpy as np

# Import custom modules
from src.data_loader import load_lfp_data, separate_by_frequency
from src.preprocessing import preprocess_pipeline
from src.outlier_detection import (
    find_outlier_indices,
    remove_outliers,
    print_removal_summary
)
from src.analysis import (
    compute_erp,
    find_erp_peak,
    compute_psd,
    compute_peak_frequency,
    compute_band_power,
    compute_band_ratio,
    analyze_band_power_statistics,
    DEFAULT_FREQ_BANDS
)
from src.visualization import (
    plot_variance_distribution,
    plot_outlier_comparison,
    plot_before_after_removal,
    plot_erp,
    plot_psd,
    plot_band_power
)


def main():
    # Configuration
    DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'mouseLFP.mat')
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Mouse LFP Analysis Pipeline")
    print("=" * 60)

    # Step 1: Load Data
    raw_sig, raw_freq = load_lfp_data(DATA_PATH)
    sig_low, sig_high, freq_info = separate_by_frequency(raw_sig, raw_freq)

    print(f"  Loaded {len(sig_low)} sessions")
    for i, (low, high) in enumerate(freq_info):
        print(f"  Session {i}: {low:.0f} Hz vs {high:.0f} Hz")

    # Step 2: Preprocessing
    processed_low = preprocess_pipeline(
        sig_low,
        original_fs=10000,
        target_fs=1000,
        lowcut=0.1,
        highcut=500,
        filter_order=4,
        baseline_start=0,
        baseline_end=100
    )

    processed_high = preprocess_pipeline(
        sig_high,
        original_fs=10000,
        target_fs=1000,
        lowcut=0.1,
        highcut=500,
        filter_order=4,
        baseline_start=0,
        baseline_end=100
    )

    print("  Preprocessing complete.")

    # Step 3: Outlier Detection and Removal

    # Find outliers
    outlier_idx_low, detection_info_low = find_outlier_indices(processed_low, k=3)
    outlier_idx_high, detection_info_high = find_outlier_indices(processed_high, k=3)

    # Visualize outlier detection
    plot_variance_distribution(
        detection_info_low,
        title_suffix="(Low Frequency)",
        save_path=os.path.join(OUTPUT_DIR, 'outlier_detection_low.png')
    )
    plot_variance_distribution(
        detection_info_high,
        title_suffix="(High Frequency)",
        save_path=os.path.join(OUTPUT_DIR, 'outlier_detection_high.png')
    )

    # Visualize normal vs outlier trials
    plot_outlier_comparison(
        processed_low,
        outlier_idx_low,
        title_suffix="(Low Frequency)",
        save_path=os.path.join(OUTPUT_DIR, 'outlier_comparison_low.png')
    )
    plot_outlier_comparison(
        processed_high,
        outlier_idx_high,
        title_suffix="(High Frequency)",
        save_path=os.path.join(OUTPUT_DIR, 'outlier_comparison_high.png')
    )

    # Remove outliers
    clean_low, summary_low = remove_outliers(processed_low, outlier_idx_low)
    clean_high, summary_high = remove_outliers(processed_high, outlier_idx_high)

    # Visualize before/after removal
    plot_before_after_removal(
        processed_low,
        clean_low,
        title_suffix="(Low Frequency)",
        save_path=os.path.join(OUTPUT_DIR, 'before_after_low.png')
    )
    plot_before_after_removal(
        processed_high,
        clean_high,
        title_suffix="(High Frequency)",
        save_path=os.path.join(OUTPUT_DIR, 'before_after_high.png')
    )

    # Print summary
    print_removal_summary(summary_low, summary_high)

    # Step 4: ERP Analysis (Method 1)
    erp_low = compute_erp(clean_low, fs=1000)
    erp_high = compute_erp(clean_high, fs=1000)

    # Plot ERPs
    plot_erp(
        erp_low,
        erp_high,
        freq_info,
        save_path=os.path.join(OUTPUT_DIR, 'erp_analysis.png')
    )

    # Print ERP peak analysis
    print("\n  ERP Peak Analysis (100-200 ms window):")
    for session_idx in range(4):
        peak_amp_low, peak_lat_low = find_erp_peak(erp_low[session_idx]['mean'])
        peak_amp_high, peak_lat_high = find_erp_peak(erp_high[session_idx]['mean'])

        print(f"\n  Session {session_idx}:")
        print(f"    Low Freq:  Peak = {peak_amp_low:.2f} at {peak_lat_low:.0f} ms")
        print(f"    High Freq: Peak = {peak_amp_high:.2f} at {peak_lat_high:.0f} ms")

    # Step 5: Frequency Analysis (Method 2)
    # Extract post-stimulus window (100-300 ms)
    analysis_start, analysis_end = 100, 300
    clean_low_post = [sig[:, analysis_start:analysis_end] for sig in clean_low]
    clean_high_post = [sig[:, analysis_start:analysis_end] for sig in clean_high]

    # Compute PSD
    psd_low = compute_psd(clean_low_post, fs=1000)
    psd_high = compute_psd(clean_high_post, fs=1000)

    plot_psd(
        psd_low,
        psd_high,
        freq_info,
        save_path=os.path.join(OUTPUT_DIR, 'psd_analysis.png')
    )

    # Peak frequency analysis
    print("\n  Peak Frequency Analysis (10-200 Hz range):")
    for session_idx in range(4):
        peak_freq_low, _ = compute_peak_frequency(psd_low[session_idx])
        peak_freq_high, _ = compute_peak_frequency(psd_high[session_idx])
        print(f"  Session {session_idx}: Low = {peak_freq_low:.1f} Hz, High = {peak_freq_high:.1f} Hz")

    # Band power analysis
    band_power_low = compute_band_power(clean_low_post, fs=1000, freq_bands=DEFAULT_FREQ_BANDS)
    band_power_high = compute_band_power(clean_high_post, fs=1000, freq_bands=DEFAULT_FREQ_BANDS)

    plot_band_power(
        band_power_low,
        band_power_high,
        freq_info,
        DEFAULT_FREQ_BANDS,
        save_path=os.path.join(OUTPUT_DIR, 'band_power_analysis.png')
    )

    # Statistical analysis
    print("\n  Band Power Statistics:")
    stats_results = analyze_band_power_statistics(band_power_low, band_power_high)

    for session_stats in stats_results:
        print(f"\n  Session {session_stats['session']}:")
        for band_name, band_stats in session_stats['bands'].items():
            sig_marker = '*' if band_stats['significant'] else ''
            print(f"    {band_name:12s}: Ratio = {band_stats['ratio']:.2f}, "
                  f"p = {band_stats['p_value']:.4f}{sig_marker}")

    # Complete
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"Figures saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
