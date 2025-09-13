# src/automatic_eq_equalizer/core/auto_eq.py

import numpy as np
import time
import queue
import copy
from PyQt5.QtCore import QThread, pyqtSignal

from ..audio_measurement.measurement import FrequencyResponseMeasurement
from ..eq_control.equalizer_apo import EqualizerPreset
from ..optimization.optimizer import (
    optimize_eq_parameters,
    compute_eq_curve
)
from .. import config
from .. import utils


class AutoEQIterative(QThread):
    """
    The core worker thread for performing iterative EQ correction.
    Manages measurement, filter design, optimization, and communication with the UI.
    """
    # Redefined signal: added target_curve (ndarray), removed target_value (float)
    update_freq_signal = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list)
    update_error_signal = pyqtSignal(int, float, float, float)

    def __init__(self):
        super(AutoEQIterative, self).__init__()
        self.baseline_response = None
        self.freqs = None
        self.target_mag = None  # Will hold the Harman target curve
        self.filter_list = []
        self.correction_factor = config.CORRECTION_FACTOR
        self.best_rms_error = float('inf')
        self.best_filter_list = []
        self.best_corrected_response = None

    def measure_response(self, start_freq, end_freq):
        # ... (This method remains unchanged) ...
        """Measure the frequency response."""
        frm = FrequencyResponseMeasurement(
            start_freq=start_freq,
            end_freq=end_freq,
            sweep_duration=config.SWEEP_DURATION,
            num_averages=config.NUM_AVERAGES
        )
        try:
            print("Measuring frequency response...")
            results = frm.run_measurement()
            return results['frequencies'], results['magnitude'], frm.sample_rate
        except Exception as e:
            print("Measurement error:", e)
            return None, None, None
        finally:
            frm.cleanup()

    def apply_filters(self):
        # ... (This method remains unchanged) ...
        """Apply all filters to Equalizer APO."""
        preset = EqualizerPreset()
        for i, (fc, gain, q) in enumerate(self.filter_list):
            print(f"Filter {i + 1}: fc={fc:.1f} Hz, gain={gain:.1f} dB, Q={q:.2f}")
            preset.add_filter(enabled=True, filter_code="PK", fc=fc, gain=gain, q=q)
        preset.apply_to_file(config.EQ_CONFIG_PATH)

    def reset_eq(self):
        # ... (This method remains unchanged) ...
        """Reset/disable all EQ filters."""
        print("Resetting equalizer...")
        preset = EqualizerPreset()
        preset.apply_to_file(config.EQ_CONFIG_PATH)
        time.sleep(0.5)

    def compute_rms_error(self, freqs, measured):
        """Compute RMS error relative to the generated target curve."""
        freqs_masked, meas_masked = utils.mask_high_freqs_target(freqs, measured)

        # Find the portion of the target curve that corresponds to the masked frequencies
        end_idx = len(freqs_masked)
        target_masked = self.target_mag[:end_idx]

        deviation = meas_masked - target_masked
        return np.sqrt(np.mean(deviation ** 2))

    def find_largest_deviation_peak(self, freqs, measured):
        """Find the largest deviation from the target curve."""
        freqs_masked, meas_masked = utils.mask_high_freqs_target(freqs, measured)

        end_idx = len(freqs_masked)
        target_masked = self.target_mag[:end_idx]

        deviation = meas_masked - target_masked
        idx = np.argmax(np.abs(deviation))
        return freqs_masked[idx], deviation[idx]

    def estimate_q_factor(self, freqs, measured, f_center, deviation, db_window=3.0):
        """Estimate a Q factor based on deviation from the target curve."""
        freqs_masked, meas_masked = utils.mask_high_freqs_target(freqs, measured)

        end_idx = len(freqs_masked)
        target_masked = self.target_mag[:end_idx]

        deviation_masked = meas_masked - target_masked

        if deviation > 0:
            threshold = deviation - db_window
            if threshold < 0: threshold = 0
            valid_indices = np.where(deviation_masked >= threshold)[0]
        else:
            threshold = deviation + db_window
            if threshold > 0: threshold = 0
            valid_indices = np.where(deviation_masked <= threshold)[0]

        if len(valid_indices) < 2: return 4.0

        f_min = freqs_masked[valid_indices[0]]
        f_max = freqs_masked[valid_indices[-1]]
        bandwidth = max(f_max - f_min, 1.0)
        q_est = f_center / bandwidth
        return max(min(q_est, 10), 0.5)

    def design_filter_for_peak(self, freqs, measured, correction_factor):
        # ... (This method remains unchanged, as it calls the updated helpers) ...
        """Identify the largest deviation and design a filter to correct it."""
        f_peak, deviation = self.find_largest_deviation_peak(freqs, measured)
        gain = -deviation * correction_factor
        q = self.estimate_q_factor(freqs, measured, f_peak, deviation)
        gain = max(min(gain, 10), -10)
        f_peak = max(min(f_peak, freqs[-1]), freqs[0])
        return (f_peak, gain, q)

    def refine_or_merge_filter(self, new_filter, freq_threshold=0.2):
        # ... (This method remains unchanged) ...
        """Merge a new filter with a close existing one."""
        (fc_new, gain_new, q_new) = new_filter
        sign_new = np.sign(gain_new)

        for i, (fc_old, gain_old, q_old) in enumerate(self.filter_list):
            sign_old = np.sign(gain_old)
            if sign_new == sign_old and sign_new != 0:
                if abs(fc_new - fc_old) / fc_old < freq_threshold:
                    merged_fc = (fc_new + fc_old) / 2.0
                    merged_gain = gain_old + gain_new
                    merged_gain = max(min(merged_gain, 10), -10)
                    merged_q = (q_old + q_new) / 2.0
                    merged_q = max(min(merged_q, 10), 0.5)
                    self.filter_list[i] = (merged_fc, merged_gain, merged_q)
                    print(f"Merged filter with existing filter {i + 1}.")
                    return True
        return False

    def run(self):
        """The main execution loop for the auto-EQ process."""
        self.reset_eq()
        measurement_end_freq = config.END_FREQ * (1 + config.MEASURE_EXTRA_RATIO)

        # --- Initial Measurement and Setup ---
        freqs, measured, fs = self.measure_response(config.START_FREQ, measurement_end_freq)
        if freqs is None:
            print("Failed to measure initial response.")
            return
        self.freqs = freqs

        # Generate the target curve
        self.target_mag = utils.generate_harman_target(self.freqs)

        idx_norm = np.argmin(np.abs(freqs - config.NORM_FREQ))
        measured_norm = measured - measured[idx_norm]
        if config.SMOOTHING_WINDOW > 1:
            window = np.ones(config.SMOOTHING_WINDOW) / config.SMOOTHING_WINDOW
            measured_norm = np.convolve(measured_norm, window, mode='same')
        measured = measured_norm

        self.baseline_response = measured.copy()
        current_rms = self.compute_rms_error(freqs, measured)
        initial_rms_for_plot = current_rms
        print(f"Initial RMS Error: {current_rms:.2f} dB")

        # --- Initial Optimization using fmin_slsqp ---
        print("Running initial EQ optimization with fmin_slsqp...")
        # Optimize towards the new target curve
        optimized_x = optimize_eq_parameters(
            freqs, measured, self.target_mag,
            start_freq=config.START_FREQ, end_freq=config.END_FREQ,
            max_filters=config.INITIAL_MAX_FILTERS
        )
        self.filter_list = [(optimized_x[3 * i], optimized_x[3 * i + 1], optimized_x[3 * i + 2]) for i in
                            range(config.INITIAL_MAX_FILTERS)]
        self.apply_filters()
        print("Initial EQ filters applied.")

        # --- Measure after Initial EQ ---
        # ... (This section remains largely unchanged) ...
        freqs_initial_eq, measured_initial_eq, _ = self.measure_response(config.START_FREQ, measurement_end_freq)
        if freqs_initial_eq is None:
            print("Measurement error after initial EQ, stopping.")
            return
        self.freqs = freqs_initial_eq

        measured_initial_eq_norm = measured_initial_eq - measured_initial_eq[idx_norm]
        if config.SMOOTHING_WINDOW > 1:
            measured_initial_eq_norm = np.convolve(measured_initial_eq_norm, window, mode='same')
        measured = measured_initial_eq_norm

        current_rms = self.compute_rms_error(self.freqs, measured)
        print(f"RMS Error after initial EQ: {current_rms:.2f} dB")

        self.best_rms_error = current_rms
        self.best_filter_list = copy.deepcopy(self.filter_list)
        self.best_corrected_response = measured.copy()

        # --- Iterative Refinement Loop ---
        iteration = 0
        while iteration < config.MAX_ITERATIONS:
            freqs_masked_plot, meas_masked_plot = utils.mask_high_freqs_plot(self.freqs, measured)

            full_eq_curve = compute_eq_curve(self.freqs, self.filter_list) if self.filter_list else np.zeros_like(
                self.freqs)

            # Emit the full target curve to the plotter
            self.update_freq_signal.emit(
                freqs_masked_plot,
                self.baseline_response[:len(freqs_masked_plot)],
                self.best_corrected_response[:len(freqs_masked_plot)],
                meas_masked_plot,
                full_eq_curve[:len(freqs_masked_plot)],
                self.target_mag[:len(freqs_masked_plot)],
                self.filter_list
            )
            print(f"Iteration {iteration + 1}, RMS Error: {current_rms:.2f} dB")
            self.update_error_signal.emit(iteration + 1, current_rms, initial_rms_for_plot, self.best_rms_error)

            if current_rms < config.ERROR_THRESHOLD:
                print("Convergence reached.")
                break

            # ... (Rest of the loop logic remains the same) ...
            previous_filter_list = copy.deepcopy(self.filter_list)
            previous_rms = current_rms

            new_filter = self.design_filter_for_peak(self.freqs, measured, self.correction_factor)
            if not self.refine_or_merge_filter(new_filter, freq_threshold=0.03):
                self.filter_list.append(new_filter)
                print("Added new filter.")

            self.apply_filters()

            freqs_after, measured_after, _ = self.measure_response(config.START_FREQ, measurement_end_freq)
            if freqs_after is None:
                print("Measurement error, stopping.")
                break

            measured_after_norm = measured_after - measured_after[idx_norm]
            if config.SMOOTHING_WINDOW > 1:
                measured_after_norm = np.convolve(measured_after_norm, window, mode='same')
            measured_after = measured_after_norm

            new_rms = self.compute_rms_error(freqs_after, measured_after)
            print(f"Post-filter RMS Error: {new_rms:.2f} dB")

            if new_rms > previous_rms:
                print("New filter degraded the response. Reverting changes and reducing correction factor.")
                self.filter_list = previous_filter_list
                self.apply_filters()
                self.correction_factor = max(self.correction_factor * 0.8, 0.1)
                current_rms = previous_rms
            else:
                print("New filter improved the response. Accepting filter and increasing correction factor slightly.")
                self.correction_factor = min(self.correction_factor * 1.1, 0.9)
                measured = measured_after
                self.freqs = freqs_after
                if new_rms < self.best_rms_error:
                    self.best_rms_error = new_rms
                    self.best_filter_list = copy.deepcopy(self.filter_list)
                    self.best_corrected_response = measured.copy()
                current_rms = new_rms

            iteration += 1

        # --- Final Update ---
        print(f"Final RMS Error: {current_rms:.2f} dB")
        freqs_masked_plot, meas_masked_plot = utils.mask_high_freqs_plot(self.freqs, measured)
        full_eq_curve = compute_eq_curve(self.freqs, self.filter_list) if self.filter_list else np.zeros_like(
            self.freqs)
        self.update_freq_signal.emit(
            freqs_masked_plot,
            self.baseline_response[:len(freqs_masked_plot)],
            self.best_corrected_response[:len(freqs_masked_plot)],
            meas_masked_plot,
            full_eq_curve[:len(freqs_masked_plot)],
            self.target_mag[:len(freqs_masked_plot)],
            self.filter_list
        )
        self.update_error_signal.emit(iteration + 1, current_rms, initial_rms_for_plot, self.best_rms_error)