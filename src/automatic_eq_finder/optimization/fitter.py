# src/automatic_eq_finder/optimization/fitter.py

import numpy as np
from scipy.optimize import minimize
from .. import utils
from .. import config

def fit_target_parameters(freqs, measured, initial_bass_gain, initial_corner_freq, initial_tilt):
    """
    Finds the target curve parameters (bass gain, corner frequency, tilt) that best
    fit the measured frequency response.
    """
    
    # Mask frequencies for fitting (similar to how we calculate RMS error)
    freqs_masked, meas_masked = utils.mask_high_freqs_target(freqs, measured)
    
    # Apply smoothing to the measurement to fit the general trend rather than peaks/dips
    meas_smoothed = utils.apply_variable_smoothing(freqs_masked, meas_masked, factor=1/3)

    def cost_function(params):
        bass_gain, corner_freq, tilt = params
        
        # Generate target curve with these parameters
        target = utils.generate_harman_target(
            freqs_masked,
            bass_gain_db=bass_gain,
            corner_freq=corner_freq,
            tilt_db_per_decade=tilt
        )
        
        # Calculate MSE
        error = meas_smoothed - target
        return np.mean(error ** 2)

    initial_guess = [initial_bass_gain, initial_corner_freq, initial_tilt]
    
    # Define bounds for the parameters (matching slider ranges roughly)
    bounds = [
        (0.0, 18.0),    # Bass Gain (dB)
        (30.0, 230.0),  # Corner Freq (Hz)
        (-15.0, 3.0)    # Tilt (dB/dec)
    ]

    result = minimize(
        cost_function,
        initial_guess,
        bounds=bounds,
        method='L-BFGS-B'
    )

    if result.success:
        return result.x
    else:
        print(f"Fitting failed: {result.message}")
        return initial_guess
