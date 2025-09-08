import numpy as np
from scipy.optimize import fmin_slsqp
from scipy.interpolate import interp1d
import logging

# === EQ CONFIGURATION ===
MAX_FILTERS = 10  # Number of filters to optimize
EQ_PARAM_MIN_GAIN = -12  # dB
EQ_PARAM_MAX_GAIN = 12  # dB
PEAK_Q_MIN = 0.5
PEAK_Q_MAX = 20

# === MEASUREMENT & TARGET CONFIGURATION ===
SMOOTHING_WINDOW = 3  # Moving average window for smoothing
TARGET_RESPONSE_FILE = None  # If provided, CSV with [frequency, magnitude], else flat

# === Logging Setup ===
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


# === Filter Functions ===
def peaking_filter_response(f, fc, gain, Q):
    """
    Model a peaking filter's response as a Lorentzian shape in dB.
    Returns the contribution (in dB) at each frequency in f.
    """
    return gain / (1 + ((f - fc) / (fc / Q)) ** 2)


def compute_eq_curve(f, filters):
    """
    Sum the contributions of all parametric filters to obtain the total EQ correction (in dB).
    """
    eq_total = np.zeros_like(f)
    for fc, gain, Q in filters:
        eq_total += peaking_filter_response(f, fc, gain, Q)
    return eq_total


def calculate_rms_error(measured, target, eq_curve=None):
    """
    RMS error between the corrected response (measured + eq_curve) and target.
    """
    if eq_curve is None:
        corrected = measured
    else:
        corrected = measured + eq_curve
    error = corrected - target
    return np.sqrt(np.mean(error ** 2))


def load_target_response(freqs, target_response_file=None):
    """
    Load the target response from file if provided; otherwise assume flat (0 dB).
    """
    if target_response_file is not None:
        data = np.genfromtxt(target_response_file, delimiter=',')
        target_freqs = data[:, 0]
        target_mag = data[:, 1]
        interp_func = interp1d(target_freqs, target_mag, fill_value="extrapolate")
        return interp_func(freqs)
    else:
        return np.zeros_like(freqs)


# === Optimization Routine using fmin_slsqp ===
def optimize_eq_parameters(freqs, measured, target, start_freq, end_freq, max_filters=MAX_FILTERS):
    """
    Optimize a set of parametric EQ filters to minimize the RMS error
    between (measured + EQ) and the target response.

    The parameter vector x consists of [fc1, gain1, Q1, ..., fcN, gainN, QN].
    """
    N = max_filters  # Number of filters
    # Initial guess for center frequencies: logarithmically spaced between START_FREQ and END_FREQ.
    fc0 = np.logspace(np.log10(start_freq), np.log10(end_freq), N)
    # Start with 0 dB gain.
    gain0 = np.zeros(N)
    # Choose an initial Q as the geometric mean of PEAK_Q_MIN and PEAK_Q_MAX.
    Q0 = np.full(N, np.sqrt(PEAK_Q_MIN * PEAK_Q_MAX))
    # Build the initial guess vector x0.
    x0 = np.zeros(3 * N)
    for i in range(N):
        x0[3 * i] = fc0[i]
        x0[3 * i + 1] = gain0[i]
        x0[3 * i + 2] = Q0[i]

    # Define bounds for each parameter.
    bounds = []
    for i in range(N):
        bounds.append((start_freq, end_freq))  # fc bounds
        bounds.append((EQ_PARAM_MIN_GAIN, EQ_PARAM_MAX_GAIN))  # gain bounds
        bounds.append((PEAK_Q_MIN, PEAK_Q_MAX))  # Q bounds

    def cost(x):
        # Convert parameter vector x into a list of filter parameters.
        filters = []
        for i in range(N):
            fc = x[3 * i]
            gain = x[3 * i + 1]
            Q = x[3 * i + 2]
            filters.append([fc, gain, Q])
        # Compute the total EQ correction.
        eq_correction = compute_eq_curve(freqs, filters)
        corrected = measured + eq_correction
        error = corrected - target
        rms = np.sqrt(np.mean(error ** 2))
        return rms

    # Run the optimization.
    optimized_x = fmin_slsqp(cost, x0, bounds=bounds, iter=100, iprint=0) # iprint=0 to suppress output
    return optimized_x