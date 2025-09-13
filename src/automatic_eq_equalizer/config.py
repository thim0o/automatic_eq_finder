# src/automatic_eq_equalizer/config.py

"""
Central configuration settings for the Automatic EQ Equalizer application.
"""

# =============================================================================
# MEASUREMENT SETTINGS
# =============================================================================
SWEEP_DURATION = 1  # seconds per sweep
NUM_AVERAGES = 1  # number of sweeps to average
START_FREQ = 35  # Hz
END_FREQ = 2000  # Hz (upper limit for correction)
MEASURE_EXTRA_RATIO = 0.5  # Measure 25% higher than END_FREQ
TARGET_FREQ_RATIO = 0.7  # Use lower 90% of frequencies (up to END_FREQ) for target/error calc
NORM_FREQ = 1000  # Frequency to normalize (0 dB)
SMOOTHING_WINDOW = 3  # Smoothing window for initial measurement

# =============================================================================
# OPTIMIZATION SETTINGS
# =============================================================================
MAX_ITERATIONS = 12  # maximum number of refinement iterations to perform
ERROR_THRESHOLD = 1.0  # target RMS error (in dB) in the masked frequency region
CORRECTION_FACTOR = 0.9  # initial fraction of the deviation to correct each iteration

# === Settings for initial optimization (fmin_slsqp) ===
INITIAL_MAX_FILTERS = 10  # Number of filters for initial optimization
INITIAL_EQ_PARAM_MIN_GAIN = -20  # dB
INITIAL_EQ_PARAM_MAX_GAIN = 20  # dB
INITIAL_PEAK_Q_MIN = 0.3
INITIAL_PEAK_Q_MAX = 20

# =============================================================================
# TARGET CURVE SETTINGS (HARMAN-LIKE)
# =============================================================================
TARGET_BASS_BOOST_DB = 8  # How many dB to boost the bass.
TARGET_TILT_DB_PER_DECADE = -1.0  # Downward slope. E.g., -1dB means 1k is 1dB louder than 10k.
TARGET_CORNER_FREQ_HZ = 105.0  # The frequency where the bass boost starts to level off.

# =============================================================================
# FILE PATHS
# =============================================================================
EQ_CONFIG_PATH = r"C:\Program Files\EqualizerAPO\config\peace.txt"