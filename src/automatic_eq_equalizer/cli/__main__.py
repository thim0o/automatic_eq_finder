import numpy as np
import time
import queue
import copy
from threading import Thread
import pyqtgraph as pg
from PyQt5 import QtWidgets

from automatic_eq_equalizer.audio_measurement.measurement import FrequencyResponseMeasurement
from automatic_eq_equalizer.eq_control.equalizer_apo import EqualizerPreset
from automatic_eq_equalizer.optimization.optimizer import (
    optimize_eq_parameters,
    compute_eq_curve,
    peaking_filter_response,
    calculate_rms_error,
    load_target_response
)

# =============================================================================
# SETTINGS (adapted from speakers/autoeq.py and speakers/fmin_slsqp_EQ.py)
# =============================================================================
SWEEP_DURATION    = 1          # seconds per sweep
NUM_AVERAGES      = 1          # number of sweeps to average
START_FREQ        = 20         # Hz
END_FREQ          = 2000      # Hz (upper limit for correction)
MAX_FILTERS       = 20         # maximum number of filters to apply (for iterative refinement)
ERROR_THRESHOLD   = 1.5        # target RMS error (in dB) in the masked frequency region
CORRECTION_FACTOR = 0.75        # initial fraction of the deviation to correct each iteration
MEASURE_EXTRA_RATIO = 0.3      # Measure 30% higher than END_FREQ
TARGET_FREQ_RATIO = 0.7       # Use lower 70% of frequencies (up to END_FREQ) for target/error calc
NORM_FREQ         = (START_FREQ + END_FREQ) / 2 # Frequency to normalize (0 dB)
SMOOTHING_WINDOW  = 3          # Smoothing window for initial measurement

# === Settings for initial optimization (consistent with fmin_slsqp_EQ_optimization.py) ===
INITIAL_MAX_FILTERS = 10       # Number of filters for initial optimization
INITIAL_EQ_PARAM_MIN_GAIN = -12  # dB
INITIAL_EQ_PARAM_MAX_GAIN = 12  # dB
INITIAL_PEAK_Q_MIN = 0.5
INITIAL_PEAK_Q_MAX = 20


# =============================================================================
# Helper: Mask the upper frequencies we want to ignore for target calculation
# =============================================================================
def mask_high_freqs_target(freqs, data, end_freq_correction, target_ratio=TARGET_FREQ_RATIO):
    """
    Ignore the highest (1 - target_ratio) portion of the frequency range
    *up to end_freq_correction* for target and error calculations.
    For example, target_ratio=0.7 ignores the top 30% of frequencies within the correction range.
    Returns (freqs_masked, data_masked).
    """
    limit_f_target = end_freq_correction * target_ratio
    idx_lim_target = np.searchsorted(freqs, limit_f_target, side='right')
    return freqs[:idx_lim_target], data[:idx_lim_target]


# =============================================================================
# Helper: Mask the upper frequencies for plotting (just up to end_freq_correction)
# =============================================================================
def mask_high_freqs_plot(freqs, data, end_freq_correction):
    """
    Mask frequencies above end_freq_correction for plotting purposes (e.g., filter design).
    Returns (freqs_masked, data_masked).
    """
    idx_lim_plot = np.searchsorted(freqs, end_freq_correction, side='right')
    return freqs[:idx_lim_plot], data[:idx_lim_plot]


# =============================================================================
# RealTimePlotter: Plot measured, baseline, and target responses plus error
# =============================================================================
class RealTimePlotter:
    def __init__(self):
        self.app = QtWidgets.QApplication([])
        self.win = pg.GraphicsLayoutWidget(title="Real-time EQ Optimization")
        self.win.resize(1200, 800)

        # Frequency response plot
        self.freq_plot = self.win.addPlot(title="Frequency Response", row=0, col=0)
        self.freq_plot.setLogMode(x=True, y=False)
        self.freq_plot.setLabels(left="Magnitude (dB)", bottom="Frequency (Hz)")
        self.freq_plot.showGrid(x=True, y=True)

        # Error over iteration plot
        self.error_plot = self.win.addPlot(title="Optimization Error", row=1, col=0)
        self.error_plot.setLabels(left="Error (dB RMS)", bottom="Iteration")
        self.error_plot.showGrid(x=True, y=True)

        # Initialize curves
        self.original_curve = self.freq_plot.plot(pen='c', name="Original") # Cyan
        self.best_corrected_curve = self.freq_plot.plot(pen='y', name="Best Corrected") # Yellow
        self.latest_corrected_curve = self.freq_plot.plot(pen='m', name="Latest Corrected") # Magenta
        self.target_line = self.freq_plot.plot(pen=pg.mkPen((192, 192, 192), width=1, style=pg.QtCore.Qt.PenStyle.DashLine), name="Target") # Light Gray Dashed
        self.error_curve = self.error_plot.plot(pen='g') # Green

        # Error bar plot
        self.error_bar_plot = self.win.addPlot(title="Summary RMS Error", row=2, col=0)
        self.error_bar_plot.setLabels(left="RMS Error (dB)")
        self.error_bar_plot.showGrid(x=False, y=True)
        self.error_bar_plot.setXRange(-0.5, 2.5) # For 3 bars at 0, 1, 2
        self.error_bar_plot.getAxis('bottom').setTicks([[(0, 'Original'), (1, 'Best'), (2, 'Latest')]])

        # Initialize bar graphs
        self.original_error_bar = pg.BarGraphItem(x=[0], height=[0], width=0.6, brush='c')
        self.best_error_bar = pg.BarGraphItem(x=[1], height=[0], width=0.6, brush='y')
        self.latest_error_bar = pg.BarGraphItem(x=[2], height=[0], width=0.6, brush='m')

        self.error_bar_plot.addItem(self.original_error_bar)
        self.error_bar_plot.addItem(self.best_error_bar)
        self.error_bar_plot.addItem(self.latest_error_bar)

        self.freq_plot.addLegend()

        # Data storage for error plot
        self.error_data = []
        self.iteration_data = []

        self.win.show()

    def update_freq_response(self, freqs, original, best_corrected, latest_corrected, target_value=None):
        """
        Plot original (blue), best corrected (green), latest corrected (red), and a target line (white dashed).
        If target_value is None, it defaults to 0 dB.
        """
        if target_value is None:
            target_value = 0
        self.original_curve.setData(freqs, original)
        self.best_corrected_curve.setData(freqs, best_corrected)
        # self.latest_corrected_curve.setData(freqs, np.atleast_1d(latest_corrected))
        self.target_line.setData(freqs, np.full_like(freqs, target_value))

    def update_error(self, iteration, error_value, original_rms, best_rms):
        """Append an error value for the current iteration and update RMS error bars."""
        self.error_data.append(error_value)
        self.iteration_data.append(iteration)
        self.error_curve.setData(self.iteration_data, self.error_data)

        # Update bar graphs
        self.original_error_bar.setOpts(height=[original_rms])
        self.best_error_bar.setOpts(height=[best_rms])
        self.latest_error_bar.setOpts(height=[error_value])

    def process_events(self):
        """Keep the GUI responsive."""
        self.app.processEvents()


# =============================================================================
# AutoEQIterative: Iterative EQ correction using filter design and merging
# =============================================================================
class AutoEQIterative:
    def __init__(self):
        self.plotter = RealTimePlotter()
        self.update_queue = queue.Queue()

        # We'll store the "baseline_response" (first measured) for plotting reference.
        self.baseline_response = None
        self.freqs = None

        # List of all filters applied, as tuples: (fc, gain, q)
        self.filter_list = []

        # Adaptive correction factor (starts at CORRECTION_FACTOR)
        self.correction_factor = CORRECTION_FACTOR

        # New attributes for tracking best performance
        self.best_rms_error = float('inf')
        self.best_filter_list = []
        self.best_corrected_response = None # To store the magnitude data for the best curve

    def measure_response(self, start_freq=START_FREQ, end_freq=END_FREQ,
                         sweep_duration=SWEEP_DURATION, num_averages=NUM_AVERAGES):
        """
        Measure the frequency response using the given sweep settings.
        """
        frm = FrequencyResponseMeasurement(
            start_freq=start_freq,
            end_freq=end_freq,
            sweep_duration=sweep_duration,
            num_averages=num_averages
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
        """
        Apply all filters in self.filter_list to Equalizer APO.
        """
        preset = EqualizerPreset()
        for i, (fc, gain, q) in enumerate(self.filter_list):
            print(f"Filter {i+1}: fc={fc:.1f} Hz, gain={gain:.1f} dB, Q={q:.2f}")
            preset.add_filter(enabled=True, filter_code="PK", fc=fc, gain=gain, q=q)
        config_path = r"C:\Program Files\EqualizerAPO\config\peace.txt" # Default path for Equalizer APO
        preset.apply_to_file(config_path)

    def reset_eq(self):
        """Reset/disable all EQ filters."""
        print("Resetting equalizer...")
        preset = EqualizerPreset()
        config_path = r"C:\Program Files\EqualizerAPO\config\peace.txt" # Default path for Equalizer APO
        preset.apply_to_file(config_path)
        time.sleep(0.5)

    def compute_rms_error(self, freqs, measured):
        """
        Compute RMS error relative to the target,
        where target is the average of the lower TARGET_FREQ_RATIO portion of the frequency range
        up to END_FREQ.
        """
        freqs_masked, meas_masked = mask_high_freqs_target(freqs, measured, END_FREQ)
        target = np.mean(meas_masked)
        deviation = meas_masked - target
        return np.sqrt(np.mean(deviation**2))

    def find_largest_deviation_peak(self, freqs, measured):
        """
        Find the frequency bin with the largest absolute deviation from the target
        (where target is the average value over the lower TARGET_FREQ_RATIO portion up to END_FREQ).
        Returns (f_peak, deviation_value) where deviation_value = measured - target.
        """
        freqs_masked, meas_masked = mask_high_freqs_target(freqs, measured, END_FREQ)
        target = np.mean(meas_masked)
        deviation = meas_masked - target
        idx = np.argmax(np.abs(deviation))
        return freqs_masked[idx], deviation[idx]

    def estimate_q_factor(self, freqs, measured, f_center, deviation, db_window=3.0):
        """
        Estimate a Q factor by measuring the bandwidth (within a ±3 dB window)
        around the deviation peak, using the masked frequency region (lower TARGET_FREQ_RATIO up to END_FREQ).
        """
        freqs_masked, meas_masked = mask_high_freqs_target(freqs, measured, END_FREQ)
        target = np.mean(meas_masked)
        deviation_masked = meas_masked - target

        if deviation > 0:
            threshold = deviation - db_window
            if threshold < 0:
                threshold = 0
            valid_indices = np.where(deviation_masked >= threshold)[0]
        else:
            threshold = deviation + db_window
            if threshold > 0:
                threshold = 0
            valid_indices = np.where(deviation_masked <= threshold)[0]

        if len(valid_indices) < 2:
            return 4.0  # fallback value

        f_min = freqs_masked[valid_indices[0]]
        f_max = freqs_masked[valid_indices[-1]]
        bandwidth = max(f_max - f_min, 1.0)
        q_est = f_center / bandwidth
        return max(min(q_est, 10), 0.5)

    def design_filter_for_peak(self, freqs, measured, correction_factor):
        """
        Identify the largest deviation (measured minus target) and design a filter to correct it.
        Correction is applied partially using the given correction_factor.
        Returns (fc, gain, q).
        """
        f_peak, deviation = self.find_largest_deviation_peak(freqs, measured)
        # Apply partial correction
        gain = -deviation * correction_factor
        q = self.estimate_q_factor(freqs, measured, f_peak, deviation)
        gain = max(min(gain, 10), -10)
        f_peak = max(min(f_peak, freqs[-1]), freqs[0])
        return (f_peak, gain, q)
    def refine_or_merge_filter(self, new_filter, freq_threshold=0.2):
        """
        If a new filter's center frequency is close to an existing filter's (within freq_threshold fraction)
        and has the same sign of gain, merge them (by averaging frequency and Q, summing gains).
        Returns True if a merge occurred.
        """
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
                    print(f"Merged filter with existing filter {i+1}.")
                    return True
        return False

    def run(self):
        # Reset the EQ first.
        self.reset_eq()

        # Calculate the end frequency for measurement (30% higher than END_FREQ for correction)
        measurement_end_freq = END_FREQ * (1 + MEASURE_EXTRA_RATIO)

        # Measure the baseline response, measuring up to measurement_end_freq.
        freqs, measured, fs = self.measure_response(
            start_freq=START_FREQ, end_freq=measurement_end_freq,
            sweep_duration=SWEEP_DURATION, num_averages=NUM_AVERAGES
        )
        if freqs is None:
            print("Failed to measure initial response.")
            return

        # 2) Normalize initial measurement (like in fmin_slsqp_EQ_optimization.py)
        idx_norm = np.argmin(np.abs(freqs - NORM_FREQ))
        measured_norm = measured - measured[idx_norm]
        if SMOOTHING_WINDOW > 1:
            window = np.ones(SMOOTHING_WINDOW) / SMOOTHING_WINDOW
            measured_norm = np.convolve(measured_norm, window, mode='same')
        measured = measured_norm # Use normalized and smoothed measurement for optimization

        self.baseline_response = measured.copy() # Store the initial normalized measurement as baseline
        current_rms = self.compute_rms_error(freqs, measured)
        initial_rms_for_plot = current_rms # Store initial RMS for bar plot
        print(f"Initial RMS Error: {current_rms:.2f} dB")

        # --- Initial Optimization using fmin_slsqp ---
        print("Running initial EQ optimization with fmin_slsqp...")
        target_mag = np.zeros_like(freqs) # Flat target for initial optimization
        optimized_x = optimize_eq_parameters(
            freqs, measured, target_mag,
            start_freq=START_FREQ, end_freq=END_FREQ,
            max_filters=INITIAL_MAX_FILTERS
        )
        initial_filters = []
        N_initial = INITIAL_MAX_FILTERS
        for i in range(N_initial):
            fc = optimized_x[3 * i]
            gain = optimized_x[3 * i + 1]
            Q = optimized_x[3 * i + 2]
            initial_filters.append((fc, gain, Q))

        self.filter_list = initial_filters
        self.apply_filters()
        print("Initial EQ filters applied.")

        # Measure response after initial EQ
        freqs_initial_eq, measured_initial_eq, _ = self.measure_response(
            start_freq=START_FREQ, end_freq=measurement_end_freq,
            sweep_duration=SWEEP_DURATION, num_averages=NUM_AVERAGES
        )
        if freqs_initial_eq is None:
            print("Measurement error after initial EQ, stopping.")
            return

        # Normalize measurement after initial EQ
        measured_initial_eq_norm = measured_initial_eq - measured_initial_eq[idx_norm]
        if SMOOTHING_WINDOW > 1:
            measured_initial_eq_norm = np.convolve(measured_initial_eq_norm, window, mode='same')
        measured = measured_initial_eq_norm # Use normalized and smoothed measurement

        current_rms = self.compute_rms_error(freqs, measured)
        print(f"RMS Error after initial EQ: {current_rms:.2f} dB")

        # Initialize best_corrected_response after initial optimization
        self.best_rms_error = current_rms
        self.best_filter_list = copy.deepcopy(self.filter_list)
        self.best_corrected_response = measured.copy()


        iteration = 0
        while iteration < MAX_FILTERS: # Use MAX_FILTERS for iterative refinement now
            # Compute the current target as the average of the *lower TARGET_FREQ_RATIO* frequencies
            # up to END_FREQ.
            freqs_masked_target, meas_masked_target = mask_high_freqs_target(freqs, measured, END_FREQ)
            target_value = np.mean(meas_masked_target)

            # Mask for plotting only up to END_FREQ
            freqs_masked_plot, meas_masked_plot = mask_high_freqs_plot(freqs, measured, END_FREQ)


            # Update the frequency response plot with original, best, and latest corrected curves.
            self.plotter.update_freq_response(
                freqs_masked_plot,
                ### self.baseline_response[:len(freqs_masked_plot)], CAUSED ERROR
                self.best_corrected_response[:len(freqs_masked_plot)],
                np.atleast_1d(meas_masked_plot), # Latest corrected response
                target_value
            )
            print(f"Iteration {iteration+1}, RMS Error: {current_rms:.2f} dB")
            self.plotter.update_error(iteration+1, current_rms, initial_rms_for_plot, self.best_rms_error)

            if current_rms < ERROR_THRESHOLD:
                print("Convergence reached.")
                break

            # Save the current filter list and RMS error for potential rollback.
            previous_filter_list = copy.deepcopy(self.filter_list)
            previous_rms = current_rms

            # Design a new filter using the adaptive correction factor.
            new_filter = self.design_filter_for_peak(freqs, measured, self.correction_factor)

            # Try merging with an existing filter.
            merged = self.refine_or_merge_filter(new_filter, freq_threshold=0.03)
            if not merged:
                self.filter_list.append(new_filter)
                print("Added new filter.")

            # Apply the filters.
            self.apply_filters()

            # Brief pause to allow EQ to update.
            # time.sleep(0.1)

            # Measure the new response, again measuring up to measurement_end_freq.
            freqs_after, measured_after, _ = self.measure_response(
                start_freq=START_FREQ, end_freq=measurement_end_freq,
                sweep_duration=SWEEP_DURATION, num_averages=NUM_AVERAGES
            )
            if freqs_after is None:
                print("Measurement error, stopping.")
                break

            # Normalize measurement after iterative EQ
            measured_after_norm = measured_after - measured_after[idx_norm]
            if SMOOTHING_WINDOW > 1:
                measured_after_norm = np.convolve(measured_after_norm, window, mode='same')
            measured_after = measured_after_norm

            new_rms = self.compute_rms_error(freqs_after, measured_after)
            print(f"Post-filter RMS Error: {new_rms:.2f} dB")

            # Validate the new filter by comparing RMS errors.
            if new_rms > previous_rms:
                # If performance worsened, rollback the change.
                print("New filter degraded the response. Reverting changes and reducing correction factor.")
                self.filter_list = previous_filter_list
                self.apply_filters()
                # Reduce the correction factor (but do not go below a minimum value).
                self.correction_factor = max(self.correction_factor * 0.8, 0.1)
                # Keep the previous measurement.
                new_rms = previous_rms
            else:
                # If performance improved, update best and increase correction factor slightly.
                print("New filter improved the response. Accepting filter and increasing correction factor slightly.")
                self.correction_factor = min(self.correction_factor * 1.1, 0.8)
                # Accept the new measurement.
                measured = measured_after
                if new_rms < self.best_rms_error:
                    self.best_rms_error = new_rms
                    self.best_filter_list = copy.deepcopy(self.filter_list)
                    self.best_corrected_response = measured.copy()

            current_rms = new_rms
            iteration += 1
            self.plotter.process_events()

        # Final update and error report.
        freqs_masked_plot, meas_masked_plot = mask_high_freqs_plot(freqs, measured, END_FREQ) #mask for final plot
        self.plotter.update_freq_response(
            freqs_masked_plot,
            self.baseline_response[:len(freqs_masked_plot)],
            self.best_corrected_response[:len(freqs_masked_plot)],
            meas_masked_plot, # Final latest corrected response
            np.full_like(freqs_masked_plot, target_value)
        )
        final_rms = self.compute_rms_error(freqs, measured)
        self.plotter.update_error(iteration+1, final_rms, initial_rms_for_plot, self.best_rms_error)
        print(f"Final RMS Error: {final_rms:.2f} dB")

        while True:
            self.plotter.process_events()
            time.sleep(0.01)


def main():
    auto_eq = AutoEQIterative()
    auto_eq.run()

if __name__ == "__main__":
    main()