# src/automatic_eq_equalizer/utils.py

"""
A module for all utility and helper functions.
"""
import os

import numpy as np
from . import config


def ensure_output_dir_exists():
    """Checks if the audio output directory exists and creates it if not."""
    # Get the path from the config file
    dir_path = config.AUDIO_OUTPUT_DIR
    # Create the directory, does nothing if it already exists
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


# --- MASKING FUNCTIONS ---


def mask_high_freqs_target(freqs, data):
    """
    Ignore the highest portion of the frequency range for target and error calculations.
    """
    limit_f_target = config.END_FREQ * config.TARGET_FREQ_RATIO
    idx_lim_target = np.searchsorted(freqs, limit_f_target, side="right")
    return freqs[:idx_lim_target], data[:idx_lim_target]


def mask_high_freqs_plot(freqs, data):
    """
    Mask frequencies above end_freq_correction for plotting purposes.
    """
    idx_lim_plot = np.searchsorted(freqs, config.END_FREQ, side="right")
    return freqs[:idx_lim_plot], data[:idx_lim_plot]


# --- SMOOTHING FUNCTION ---


def apply_variable_smoothing(
    freqs: np.ndarray, data: np.ndarray, factor: float = 1 / 6
) -> np.ndarray:
    """
    Applies variable (fractional-octave) smoothing to frequency response data.
    This is crucial for making the optimizer focus on broad tonal issues.
    """
    smoothed_data = np.zeros_like(data)
    for i, f in enumerate(freqs):
        # Define the frequency window for averaging
        f_lower = f / (2 ** (factor / 2))
        f_upper = f * (2 ** (factor / 2))
        indices = np.where((freqs >= f_lower) & (freqs <= f_upper))

        if indices[0].size > 0:
            smoothed_data[i] = np.mean(data[indices])
        else:
            smoothed_data[i] = data[i]

    return smoothed_data


# --- TARGET CURVE GENERATION FUNCTION ---


def generate_harman_target(
    freqs: np.ndarray,
    bass_gain_db: float = config.TARGET_BASS_BOOST_DB,
    corner_freq: float = config.TARGET_CORNER_FREQ_HZ,
    tilt_db_per_decade: float = config.TARGET_TILT_DB_PER_DECADE,
) -> np.ndarray:
    """
    Generates a Harman-like target curve with a customizable low-shelf bass boost
    and a subtle downward tilt at higher frequencies.
    """
    if freqs is None or len(freqs) == 0:
        return np.array([])

    low_shelf = bass_gain_db / np.sqrt(1 + (freqs / corner_freq) ** 4)

    tilt_ref_freq = 1000.0
    tilt = tilt_db_per_decade * np.log10(freqs / tilt_ref_freq)

    target_db = low_shelf + tilt

    val_at_1k_ref = (bass_gain_db / np.sqrt(1 + (1000.0 / corner_freq) ** 4)) + (
        tilt_db_per_decade * np.log10(1000.0 / tilt_ref_freq)
    )

    normalized_target_db = target_db - val_at_1k_ref
    return normalized_target_db



def load_calibration_data(file_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads and parses microphone calibration data from a text file.
    The file is expected to have frequency (Hz) and magnitude (dB) columns,
    skipping the first two header lines.
    
    Returns:
        tuple[np.ndarray, np.ndarray]: (frequencies, magnitudes_db)
    """
    if not os.path.exists(file_path):
        print(f"Warning: Calibration file not found at {file_path}")
        return np.array([]), np.array([])

    try:
        # Load data, skipping the first two lines (header/metadata)
        # The file uses whitespace separation, which np.loadtxt handles by default.
        data = np.loadtxt(file_path, skiprows=2)
        
        if data.ndim != 2 or data.shape[1] < 2:
            print(f"Error: Calibration file {file_path} has unexpected format.")
            return np.array([]), np.array([])

        frequencies = data[:, 0]
        magnitudes_db = data[:, 1]
        
        return frequencies, magnitudes_db
        
    except Exception as e:
        print(f"Error loading calibration data from {file_path}: {e}")
        return np.array([]), np.array([])
