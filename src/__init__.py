from .data_loader import load_lfp_data, separate_by_frequency
from .preprocessing import design_bandpass_filter, apply_filter, downsample, baseline_correction
from .outlier_detection import detect_outliers_mad, remove_outliers
from .analysis import compute_erp, compute_band_power, compute_peak_frequency, compute_band_ratio
from .visualization import (
    plot_variance_distribution,
    plot_outlier_comparison,
    plot_before_after_removal,
    plot_erp,
    plot_psd,
    plot_band_power,
    plot_band_ratio
)

__version__ = "1.0.0"
__author__ = "Choi Joung Woo"
