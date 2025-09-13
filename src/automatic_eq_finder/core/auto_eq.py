# src/automatic_eq_equalizer/core/auto_eq.py

import numpy as np
import time
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
    The core worker thread for performing the auto-EQ process.
    This version uses a HYBRID approach:
    1. A fast initial optimization using the sequential method.
    2. An optional iterative fine-tuning loop for final adjustments.
    """
    update_freq_signal = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list)
    update_error_signal = pyqtSignal(int, float, float, float)

    def __init__(self):
        super(AutoEQIterative, self).__init__()
        self.baseline_response = None
        self.freqs = None
        self.target_mag = None
        self.filter_list = []
        self.correction_factor = config.CORRECTION_FACTOR
        self.best_rms_error = float('inf')
        self.best_filter_list = []
        self.best_corrected_response = None

    # --- Helper Methods (Measurement, EQ Application) ---

    def measure_response(self, start_freq, end_freq):
        # ... (This method is unchanged) ...
        frm = FrequencyResponseMeasurement(
            start_freq=start_freq, end_freq=end_freq,
            sweep_duration=config.SWEEP_DURATION, num_averages=config.NUM_AVERAGES)
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
        # ... (This method is unchanged) ...
        preset = EqualizerPreset()
        for i, (fc, gain, q) in enumerate(self.filter_list):
            print(f"Filter {i + 1}: fc={fc:.1f} Hz, gain={gain:.1f} dB, Q={q:.2f}")
            preset.add_filter(enabled=True, filter_code="PK", fc=fc, gain=gain, q=q)
        preset.apply_to_file(config.EQ_CONFIG_PATH)

    def reset_eq(self):
        # ... (This method is unchanged) ...
        print("Resetting equalizer...")
        preset = EqualizerPreset()
        preset.apply_to_file(config.EQ_CONFIG_PATH)
        time.sleep(0.5)

    def compute_rms_error(self, freqs, measured):
        # ... (This method is unchanged) ...
        freqs_masked, meas_masked = utils.mask_high_freqs_target(freqs, measured)
        end_idx = len(freqs_masked)
        target_masked = self.target_mag[:end_idx]
        deviation = meas_masked - target_masked
        return np.sqrt(np.mean(deviation ** 2))

    # --- Helper Methods for Fine-Tuning (Brought back from earlier version) ---

    def find_largest_deviation_peak(self, freqs, measured):
        """Finds the largest deviation from the target curve."""
        freqs_masked, meas_masked = utils.mask_high_freqs_target(freqs, measured)
        target_masked = self.target_mag[:len(freqs_masked)]
        deviation = meas_masked - target_masked
        idx = np.argmax(np.abs(deviation))
        return freqs_masked[idx], deviation[idx]

    def design_filter_for_peak(self, freqs, measured, correction_factor):
        """Designs a single filter to correct the largest remaining peak."""
        f_peak, deviation = self.find_largest_deviation_peak(freqs, measured)
        gain = -deviation * correction_factor
        # A simple Q estimation for refinement is often sufficient
        q = 2.0
        gain = np.clip(gain, -10, 10)
        f_peak = np.clip(f_peak, freqs[0], freqs[-1])
        return (f_peak, gain, q)

    def refine_or_merge_filter(self, new_filter, freq_threshold=0.1):
        """Merges a new filter with a close existing one, if applicable."""
        (fc_new, gain_new, q_new) = new_filter
        sign_new = np.sign(gain_new)
        for i, (fc_old, gain_old, q_old) in enumerate(self.filter_list):
            if abs(fc_new - fc_old) / fc_old < freq_threshold:
                merged_gain = gain_old + gain_new
                merged_gain = np.clip(merged_gain, config.INITIAL_EQ_PARAM_MIN_GAIN, config.INITIAL_EQ_PARAM_MAX_GAIN)
                self.filter_list[i] = (fc_old, merged_gain, q_old)
                print(f"Merged new correction with existing filter at {fc_old:.1f} Hz.")
                return True
        return False

    # --- THE NEW HYBRID `run` METHOD ---

        # src/automatic_eq_equalizer/core/auto_eq.py

    def run(self):
        self.reset_eq()
        measurement_end_freq = config.END_FREQ * (1 + config.MEASURE_EXTRA_RATIO)

        # --- 1. Initial Measurement & IMMEDIATE UI Update ---
        freqs, measured, fs = self.measure_response(config.START_FREQ, measurement_end_freq)
        if freqs is None: return
        self.freqs = freqs
        self.target_mag = utils.generate_harman_target(self.freqs)

        idx_norm = np.argmin(np.abs(freqs - config.NORM_FREQ))
        measured_norm = measured - measured[idx_norm]
        self.baseline_response = np.copy(measured_norm)

        initial_rms_for_plot = self.compute_rms_error(freqs, self.baseline_response)
        print(f"Initial RMS Error (unsmoothed): {initial_rms_for_plot:.2f} dB")

        # <<< NEW: UPDATE UI IMMEDIATELY AFTER FIRST RUN >>>
        freqs_masked_plot, _ = utils.mask_high_freqs_plot(self.freqs, self.baseline_response)
        # Initialize best_corrected_response here to avoid issues if the loop doesn't run
        self.best_corrected_response = np.copy(self.baseline_response)
        self.update_freq_signal.emit(
            freqs_masked_plot,
            self.baseline_response[:len(freqs_masked_plot)],
            self.best_corrected_response[:len(freqs_masked_plot)],  # Best is same as original
            self.baseline_response[:len(freqs_masked_plot)],  # Latest is same as original
            np.zeros_like(freqs_masked_plot),  # No EQ curve yet
            self.target_mag[:len(freqs_masked_plot)],
            []  # No filters yet
        )
        self.update_error_signal.emit(0, initial_rms_for_plot, initial_rms_for_plot, initial_rms_for_plot)
        # <<< END OF IMMEDIATE UPDATE >>>

        # --- 2. Main Sequential Optimization ---
        measured_for_optimizer = utils.apply_variable_smoothing(
            self.freqs, measured_norm, factor=config.OPTIMIZER_SMOOTHING_FACTOR)

        optimized_x = optimize_eq_parameters(
            self.freqs, measured_for_optimizer, self.target_mag,
            start_freq=config.START_FREQ, end_freq=config.END_FREQ,
            max_filters=config.INITIAL_MAX_FILTERS)

        self.filter_list = [(optimized_x[3 * i], optimized_x[3 * i + 1], optimized_x[3 * i + 2]) for i in
                            range(config.INITIAL_MAX_FILTERS)]

        self.filter_list.sort(key=lambda f: f[0]) # sort on freq for better readability

        self.apply_filters()
        print("Initial sequential optimization applied. Starting fine-tuning loop...")

        # --- 3. Iterative Fine-Tuning Loop ---
        current_measurement = None
        iteration = 0
        while iteration < config.MAX_FINETUNE_ITERATIONS:
            # Measure the current state of the room with the latest filters
            freqs_iter, measured_iter, _ = self.measure_response(config.START_FREQ, measurement_end_freq)
            if freqs_iter is None: break

            self.freqs = freqs_iter
            current_measurement = measured_iter - measured_iter[idx_norm]
            current_rms = self.compute_rms_error(self.freqs, current_measurement)

            # Update the "best" result if this is an improvement
            if current_rms < self.best_rms_error:
                self.best_rms_error = current_rms
                self.best_filter_list = copy.deepcopy(self.filter_list)
                self.best_corrected_response = np.copy(current_measurement)

            # Update UI with the latest measurement
            freqs_masked_plot, meas_masked_plot = utils.mask_high_freqs_plot(self.freqs, current_measurement)
            full_eq_curve = compute_eq_curve(self.freqs, self.filter_list)
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

            print(
                f"Fine-tune Iteration {iteration + 1}, Current RMS: {current_rms:.2f} dB, Best RMS: {self.best_rms_error:.2f} dB")

            if current_rms < config.ERROR_THRESHOLD:
                print("Convergence threshold reached.")
                break

            # Design one more small correction
            new_filter = self.design_filter_for_peak(self.freqs, current_measurement, self.correction_factor)

            # Try to merge this correction with an existing filter, or add it if not possible
            if not self.refine_or_merge_filter(new_filter):
                self.filter_list.append(new_filter)
                print("Added new filter for fine-tuning.")

            # Apply the newly updated filter list
            self.apply_filters()
            iteration += 1

        print(f"Fine-tuning complete. Final Best RMS Error: {self.best_rms_error:.2f} dB")
        # Apply the best filter set found during fine-tuning
        self.filter_list = self.best_filter_list
        self.apply_filters()
        print("Best filter set applied.")

        # --- 4. FINAL UI CLEANUP (THE FIX) ---
        # After everything is done, send one last update to the UI to ensure
        # the "Best" and "Latest" curves both show the definitive best result.
        print("Sending final update to UI.")
        final_eq_curve = compute_eq_curve(self.freqs, self.best_filter_list)
        freqs_masked_plot, _ = utils.mask_high_freqs_plot(self.freqs, self.best_corrected_response)

        self.update_freq_signal.emit(
            freqs_masked_plot,
            self.baseline_response[:len(freqs_masked_plot)],
            self.best_corrected_response[:len(freqs_masked_plot)],  # Send BEST data
            self.best_corrected_response[:len(freqs_masked_plot)],  # Send BEST data again for latest
            final_eq_curve[:len(freqs_masked_plot)],
            self.target_mag[:len(freqs_masked_plot)],
            self.best_filter_list
        )
        # Also ensure the error bar for "Latest" matches the "Best"
        self.update_error_signal.emit(iteration + 1, self.best_rms_error, initial_rms_for_plot, self.best_rms_error)