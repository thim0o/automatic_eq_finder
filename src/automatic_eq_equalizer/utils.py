# src/automatic_eq_equalizer/utils.py

"""
Utility functions for data manipulation, such as frequency masking.
"""

import numpy as np
from . import config

def mask_high_freqs_target(freqs, data):
    """
    Ignore the highest portion of the frequency range for target and error calculations.
    Returns (freqs_masked, data_masked).
    """
    limit_f_target = config.END_FREQ * config.TARGET_FREQ_RATIO
    idx_lim_target = np.searchsorted(freqs, limit_f_target, side='right')
    return freqs[:idx_lim_target], data[:idx_lim_target]


def mask_high_freqs_plot(freqs, data):
    """
    Mask frequencies above end_freq_correction for plotting purposes.
    Returns (freqs_masked, data_masked).
    """
    idx_lim_plot = np.searchsorted(freqs, config.END_FREQ, side='right')
    return freqs[:idx_lim_plot], data[:idx_lim_plot]



# src/automatic_eq_equalizer/target_curve.py

def generate_harman_target(freqs: np.ndarray) -> np.ndarray:
    """
    Generates a Harman-like target curve with a customizable low-shelf bass boost
    and a subtle downward tilt at higher frequencies.

    This curve's shape is determined by two main configurable parameters
    from the config.py file:

    1.  TARGET_BASS_BOOST_DB: Controls "how bass heavy" the curve is.
        This sets the maximum gain (in dB) in the low-frequency region.

    2.  TARGET_CORNER_FREQ_HZ: Controls the "bass frequency".
        This sets the corner frequency of the low-shelf filter, effectively
        determining the point where the bass boost begins to level off.

    Args:
        freqs: A NumPy array of frequency points.

    Returns:
        A NumPy array of the target magnitude in dB for each frequency.
    """
    if freqs is None or len(freqs) == 0:
        return np.array([])

    # --- 1. Low-Shelf Filter for Bass Boost ---
    # This section is controlled by the two main configurable parameters.

    # Parameter 1: How bass heavy is the curve?
    bass_gain_db = config.TARGET_BASS_BOOST_DB

    # Parameter 2: The frequency where the bass boost is centered.
    corner_freq = config.TARGET_CORNER_FREQ_HZ

    # We use a 2nd-order (12 dB/octave) low-shelf filter formula. This provides a
    # smooth, natural-sounding transition into the bass region.
    # The magnitude response of the shelf is calculated here.
    low_shelf = bass_gain_db / np.sqrt(1 + (freqs / corner_freq)**4)

    # --- 2. Downward Tilt for High Frequencies ---
    # This adds a subtle, pleasant roll-off to the high frequencies.
    # This part is controlled by TARGET_TILT_DB_PER_DECADE.
    tilt_ref_freq = 1000.0  # Tilt is calculated relative to 1 kHz
    tilt = config.TARGET_TILT_DB_PER_DECADE * np.log10(freqs / tilt_ref_freq)

    # --- 3. Combine and Normalize the Curve ---
    # Combine the bass shelf and the high-frequency tilt.
    target_db = low_shelf + tilt

    # Finally, we normalize the entire curve so it passes through 0 dB at 1000 Hz.
    # This is crucial because it provides a consistent reference point, preventing
    # the entire volume from being unintentionally raised or lowered.
    val_at_1k_ref = (bass_gain_db / np.sqrt(1 + (1000.0 / corner_freq)**4)) + \
                    (config.TARGET_TILT_DB_PER_DECADE * np.log10(1000.0 / tilt_ref_freq))

    normalized_target_db = target_db - val_at_1k_ref

    return normalized_target_db