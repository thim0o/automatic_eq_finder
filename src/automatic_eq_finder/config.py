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
END_FREQ = 1000  # Hz (upper limit for correction)
MEASURE_EXTRA_RATIO = 0.55  # Measure 25% higher than END_FREQ
TARGET_FREQ_RATIO = 1  # Use lower 70% of frequencies (up to END_FREQ) for target/error calc
NORM_FREQ = 1000  # Frequency to normalize (0 dB)

# =============================================================================
# OPTIMIZATION SETTINGS
# =============================================================================
ERROR_THRESHOLD = 1.0  # target RMS error (in dB) in the masked frequency region
CORRECTION_FACTOR = 1  # initial fraction of the deviation to correct each iteration

# === Settings for initial optimization (fmin_slsqp) ===
INITIAL_MAX_FILTERS = 15  # Number of filters for initial optimization
INITIAL_EQ_PARAM_MIN_GAIN = -26  # dB
INITIAL_EQ_PARAM_MAX_GAIN = 15  # dB
INITIAL_PEAK_Q_MIN = 0.2
INITIAL_PEAK_Q_MAX = 50

# =============================================================================
# TARGET CURVE SETTINGS (HARMAN-LIKE)
# =============================================================================
TARGET_BASS_BOOST_DB = 9  # How many dB to boost the bass.
TARGET_TILT_DB_PER_DECADE = -1.0  # Downward slope. E.g., -1dB means 1k is 1dB louder than 10k.
TARGET_CORNER_FREQ_HZ = 105.0  # The frequency where the bass boost starts to level off.

# =============================================================================
# FILE PATHS
# =============================================================================
EQ_CONFIG_PATH = r"C:\Program Files\EqualizerAPO\config\peace.txt"



OPTIMIZER_SMOOTHING_FACTOR = 1/6
MAX_FINETUNE_ITERATIONS = 10

