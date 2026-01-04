# src/automatic_eq_finder/optimization/responsiveness_analyzer.py

import numpy as np
from .optimizer import compute_eq_curve

def analyze_responsiveness(freqs, previous_measurement, current_measurement, previous_filters, current_filters):
    """
    Analyzes how responsive each frequency is to the applied EQ changes.
    
    Responsiveness = (Actual Change in Response) / (Predicted Change from EQ)
    
    A value of 1.0 means the room/speaker responded exactly as predicted.
    A value near 0.0 means the EQ had little effect (e.g., a deep null).
    Values > 1.0 might indicate resonance or measurement noise.
    """
    if previous_measurement is None or current_measurement is None:
        return np.ones_like(freqs)

    # Calculate the EQ curves for both states
    prev_eq = compute_eq_curve(freqs, previous_filters)
    curr_eq = compute_eq_curve(freqs, current_filters)
    
    predicted_change = curr_eq - prev_eq
    actual_change = current_measurement - previous_measurement
    
    # To reduce noise, we apply some smoothing to the changes before calculating the ratio
    from ..utils import apply_variable_smoothing
    
    # Use a relatively fine smoothing for the inputs to preserve frequency detail
    # but remove measurement jitter.
    smooth_p = apply_variable_smoothing(freqs, predicted_change, factor=1/12)
    smooth_a = apply_variable_smoothing(freqs, actual_change, factor=1/12)
    
    # We use a regularized estimator to calculate responsiveness.
    # This avoids division by zero and naturally handles frequencies with small changes
    # by biasing the result towards 1.0 (perfect responsiveness) when data is sparse.
    # Formula: (Actual * Predicted + Alpha) / (Predicted^2 + Alpha)
    # where Alpha represents our "confidence" in the measurement.
    
    alpha = 0.1 # A value of 0.1 means we need ~0.3dB of predicted change to overcome the bias
    
    # Calculate responsiveness
    # We use the product in the numerator to ensure it works for both boosts and cuts
    responsiveness = (smooth_a * smooth_p + alpha) / (smooth_p**2 + alpha)
    
    # Clip to reasonable bounds to avoid extreme values from noise
    responsiveness = np.clip(responsiveness, 0.0, 2.0)
    
    # Final smoothing pass for display purposes
    responsiveness = apply_variable_smoothing(freqs, responsiveness, factor=1/3)
    
    return responsiveness
