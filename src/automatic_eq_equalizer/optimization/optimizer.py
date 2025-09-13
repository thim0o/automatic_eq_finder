# src/automatic_eq_equalizer/optimization/optimizer.py

import numpy as np
from scipy.optimize import fmin_slsqp
from scipy.interpolate import interp1d
import logging
from .. import config

# === Logging Setup ===
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


# === Filter Functions ===
def peaking_filter_response(f, fc, gain, Q):
    """
    Model a peaking filter's response. A simple Gaussian-like shape is often
    more stable for optimizers than a true-to-spec IIR filter model.
    """
    width = fc / Q
    return gain * np.exp(-0.5 * ((f - fc) / width) ** 2)


def compute_eq_curve(f, filters):
    """
    Sum the contributions of all parametric filters to obtain the total EQ correction (in dB).
    """
    eq_total = np.zeros_like(f)
    for fc, gain, Q in filters:
        eq_total += peaking_filter_response(f, fc, gain, Q)
    return eq_total


# === Optimization Routine using fmin_slsqp ===
def optimize_eq_parameters(freqs, measured, target, start_freq, end_freq, max_filters=config.INITIAL_MAX_FILTERS):
    """
    Optimize a set of parametric EQ filters to minimize the RMS error
    between (measured + EQ) and the target response.

    This version uses a "smart" initial guess for filter gains.
    """
    N = max_filters

    # --- Create a "Smart" Initial Guess (x0) ---

    # 1. Frequencies (fc0): Logarithmically spaced.
    # THIS IS THE CORRECTED LINE:
    fc0 = np.logspace(np.log10(start_freq), np.log10(end_freq), N)

    # 2. Gains (gain0): Make an initial guess based on the problem.
    print("Generating smart initial guess for optimizer...")
    interp_measured = interp1d(freqs, measured, kind='linear', fill_value="extrapolate")
    interp_target = interp1d(freqs, target, kind='linear', fill_value="extrapolate")
    deviation_at_fc0 = interp_measured(fc0) - interp_target(fc0)
    gain0 = -deviation_at_fc0
    gain0 = np.clip(gain0, config.INITIAL_EQ_PARAM_MIN_GAIN, config.INITIAL_EQ_PARAM_MAX_GAIN)

    # 3. Q (Q0): A middle-of-the-road Q is a good starting point.
    Q0 = np.full(N, np.sqrt(config.INITIAL_PEAK_Q_MIN * config.INITIAL_PEAK_Q_MAX))

    # 4. Build the initial guess vector x0.
    x0 = np.zeros(3 * N)
    for i in range(N):
        x0[3 * i] = fc0[i]
        x0[3 * i + 1] = gain0[i]
        x0[3 * i + 2] = Q0[i]

    # Define bounds for each parameter.
    bounds = []
    for i in range(N):
        bounds.append((start_freq, end_freq))
        bounds.append((config.INITIAL_EQ_PARAM_MIN_GAIN, config.INITIAL_EQ_PARAM_MAX_GAIN))
        bounds.append((config.INITIAL_PEAK_Q_MIN, config.INITIAL_PEAK_Q_MAX))

    def cost(x):
        """The cost function to be minimized by SLSQP."""
        filters = [(x[3 * i], x[3 * i + 1], x[3 * i + 2]) for i in range(N)]
        eq_correction = compute_eq_curve(freqs, filters)
        mask = (freqs >= start_freq) & (freqs <= end_freq)
        error = (measured[mask] + eq_correction[mask]) - target[mask]
        return np.sqrt(np.mean(error ** 2))

    # Run the optimization.
    optimized_x = fmin_slsqp(cost, x0, bounds=bounds, iter=100, iprint=0)

    return optimized_x