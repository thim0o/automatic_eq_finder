# src/automatic_eq_equalizer/optimization/optimizer.py

import numpy as np
from scipy.optimize import minimize
from .. import config

# === Logging (Optional, but good practice) ===
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# === Filter Functions (Can be simplified) ===
def peaking_filter_response(f, fc, gain, Q):
    """
    Models a single peaking filter's response. Using a Gaussian shape is stable
    and effective for optimization purposes.
    """
    if Q == 0 or fc == 0: return np.zeros_like(f)
    width = fc / Q
    return gain * np.exp(-0.5 * ((f - fc) / width) ** 2)


def compute_eq_curve(f, filters):
    """
    Sums the contributions of all parametric filters.
    """
    eq_total = np.zeros_like(f)
    if not filters: return eq_total
    for fc, gain, Q in filters:
        eq_total += peaking_filter_response(f, fc, gain, Q)
    return eq_total


# === NEW SEQUENTIAL OPTIMIZER ===
def optimize_eq_parameters(freqs, measured, target, start_freq, end_freq, max_filters=config.INITIAL_MAX_FILTERS):
    """
    Generates a set of EQ filters using a sequential ("greedy") optimization strategy.

    At each step, it identifies the largest remaining error and designs a single
    filter to best correct it, then adds that filter to the list and repeats.
    """
    print("Starting sequential optimization...")

    # Create a mask to focus the optimization on the relevant frequency range
    opt_mask = (freqs >= start_freq) & (freqs <= end_freq)

    # List to hold the filters we design
    final_filters = []

    # This is the "corrected" measurement, which we will update after adding each filter
    current_corrected_response = np.copy(measured)

    for i in range(max_filters):
        # 1. Calculate the current error curve
        current_error = current_corrected_response[opt_mask] - target[opt_mask]

        # 2. Find the single worst problem (largest deviation)
        max_error_idx = np.argmax(np.abs(current_error))
        f_peak = freqs[opt_mask][max_error_idx]
        deviation = current_error[max_error_idx]

        print(f"Filter {i + 1}/{max_filters}: Targeting peak error of {deviation:.2f} dB at {f_peak:.1f} Hz")

        # 3. Define the Cost Function for optimizing just ONE filter
        def cost_function_for_one_filter(params):
            fc, gain, Q = params

            # Create the EQ curve for just this *one* new filter
            new_filter_curve = peaking_filter_response(freqs, fc, gain, Q)

            # Hypothetical response if we added this new filter
            hypothetical_response = current_corrected_response + new_filter_curve

            # Calculate the RMS error of this hypothetical response
            error = hypothetical_response[opt_mask] - target[opt_mask]
            return np.sqrt(np.mean(error ** 2))

        # 4. Make a smart initial guess for this one filter
        initial_guess = [
            f_peak,  # fc: start at the problem frequency
            -deviation,  # gain: start with the exact opposite of the error
            2.0  # Q: start with a moderate Q value
        ]

        # Clip the initial gain guess to stay within bounds
        initial_guess[1] = np.clip(initial_guess[1], config.INITIAL_EQ_PARAM_MIN_GAIN, config.INITIAL_EQ_PARAM_MAX_GAIN)

        # 5. Define the bounds for this one filter
        bounds = [
            (start_freq, end_freq),  # fc bounds
            (config.INITIAL_EQ_PARAM_MIN_GAIN, config.INITIAL_EQ_PARAM_MAX_GAIN),  # gain bounds
            (config.INITIAL_PEAK_Q_MIN, config.INITIAL_PEAK_Q_MAX)  # Q bounds
        ]

        # 6. Run the optimization for this single filter
        result = minimize(
            cost_function_for_one_filter,
            initial_guess,
            bounds=bounds,
            method='L-BFGS-B'  # A robust and efficient solver for simple problems
        )

        # Extract the optimized parameters for the new filter
        optimized_params = result.x
        fc_opt, gain_opt, q_opt = optimized_params

        # 7. "Bake in" the new filter
        print(f"  -> Designed Filter: fc={fc_opt:.1f}, gain={gain_opt:.2f}, Q={q_opt:.2f}")
        final_filters.append((fc_opt, gain_opt, q_opt))

        # Update the corrected response with the effect of the new filter
        # This becomes the baseline for the *next* iteration
        newly_added_curve = peaking_filter_response(freqs, fc_opt, gain_opt, q_opt)
        current_corrected_response += newly_added_curve

    # 8. Convert the final filter list into the flat array format expected by auto_eq.py
    optimized_x = np.array(final_filters).flatten()

    print("Sequential optimization complete.")
    return optimized_x