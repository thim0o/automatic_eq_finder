# src/automatic_eq_equalizer/ui/realtime_plotter.py

"""Real-time plotting utilities for Automatic EQ Equalizer."""

import numpy as np
import pyqtgraph as pg


class RealTimePlotter:
    """Plot measured, baseline, and target responses plus error."""

    def __init__(self, app):
        self.app = app
        self.win = pg.GraphicsLayoutWidget(title="Real-time EQ Optimization")
        self.win.showMaximized()

        # Frequency response plot
        self.freq_plot = self.win.addPlot(title="Frequency Response", row=0, col=0)
        self.freq_plot.setLogMode(x=True, y=False)
        self.freq_plot.setLabels(left="Magnitude (dB)", bottom="Frequency (Hz)")
        self.freq_plot.showGrid(x=True, y=True)
        self.freq_plot.getAxis('bottom').setGrid(150)
        self.freq_plot.getAxis('left').setGrid(150)

        # Error over iteration plot
        self.error_plot = self.win.addPlot(title="Optimization Error", row=1, col=0)
        self.error_plot.setLabels(left="Error (dB RMS)", bottom="Iteration")
        self.error_plot.showGrid(x=True, y=True)

        # Initialize curves
        self.original_curve = self.freq_plot.plot(pen='c', name="Original")  # Cyan
        self.best_corrected_curve = self.freq_plot.plot(pen='y', name="Best Corrected")  # Yellow
        self.latest_corrected_curve = self.freq_plot.plot(pen='m', name="Latest Corrected")  # Magenta
        self.eq_profile_curve = self.freq_plot.plot(
            pen=pg.mkPen('w', width=2, style=pg.QtCore.Qt.PenStyle.DashLine),
            name="EQ Profile"
        )
        # New curve for the Harman-style target
        self.target_curve = self.freq_plot.plot(
            pen=pg.mkPen('g', width=2, style=pg.QtCore.Qt.PenStyle.SolidLine),
            name="Target Curve",
        )
        self.error_curve = self.error_plot.plot(pen='g', symbol='o')

        # Error bar plot
        self.error_bar_plot = self.win.addPlot(title="Summary RMS Error", row=2, col=0)
        self.error_bar_plot.setLabels(left="RMS Error (dB)")
        self.error_bar_plot.showGrid(x=False, y=True)
        self.error_bar_plot.setXRange(-0.5, 2.5)
        self.error_bar_plot.getAxis('bottom').setTicks([[(0, 'Original'), (1, 'Best'), (2, 'Latest')]])

        # Initialize bar graphs
        self.original_error_bar = pg.BarGraphItem(x=[0], height=[0], width=0.6, brush='c')
        self.best_error_bar = pg.BarGraphItem(x=[1], height=[0], width=0.6, brush='y')
        self.latest_error_bar = pg.BarGraphItem(x=[2], height=[0], width=0.6, brush='m')

        self.error_bar_plot.addItem(self.original_error_bar)
        self.error_bar_plot.addItem(self.best_error_bar)
        self.error_bar_plot.addItem(self.latest_error_bar)

        self.freq_plot.addLegend()

        self.error_data = []
        self.iteration_data = []
        self.filter_visuals = []

        self.win.show()

    def _update_filter_visuals(self, freqs, target_curve_data, filter_list):
        """Clear old filter visuals and draw new ones relative to the target curve."""
        for item in self.filter_visuals:
            self.freq_plot.removeItem(item)
        self.filter_visuals.clear()

        if freqs is None or len(freqs) == 0:
            return

        for fc, gain, q in filter_list:
            if gain == 0:
                continue

            # Find the target level at the filter's center frequency
            # np.interp is used to find the value even if fc isn't exactly in freqs
            target_level_at_fc = np.interp(fc, freqs, target_curve_data)

            pen_color = pg.mkPen('r' if gain < 0 else (255, 165, 0), width=1.5) # Red for cut, Orange for boost

            # Vertical line showing gain relative to the target curve
            gain_line = pg.PlotDataItem(
                x=[fc, fc],
                y=[target_level_at_fc, target_level_at_fc + gain],
                pen=pen_color
            )
            self.freq_plot.addItem(gain_line)
            self.filter_visuals.append(gain_line)

            # Horizontal line showing approximate bandwidth (Q)
            bandwidth = fc / q
            q_line = pg.PlotDataItem(
                x=[fc - bandwidth / 2, fc + bandwidth / 2],
                y=[target_level_at_fc + gain, target_level_at_fc + gain],
                pen=pen_color
            )
            self.freq_plot.addItem(q_line)
            self.filter_visuals.append(q_line)

    def update_freq_response(self, freqs, original, best_corrected, latest_corrected, eq_curve, target_curve_data, filter_list):
        """Update frequency response curves, target curve, and filter visuals."""
        self.original_curve.setData(freqs, original)
        self.best_corrected_curve.setData(freqs, best_corrected)
        self.latest_corrected_curve.setData(freqs, latest_corrected)
        self.eq_profile_curve.setData(freqs, eq_curve)
        self.target_curve.setData(freqs, target_curve_data)

        # Update the filter visualizations on the plot
        self._update_filter_visuals(freqs, target_curve_data, filter_list)

    def update_error(self, iteration, error_value, original_rms, best_rms):
        """Append an error value and update RMS error bars."""
        # ... (This method remains unchanged) ...
        self.error_data.append(error_value)
        self.iteration_data.append(iteration)
        self.error_curve.setData(self.iteration_data, self.error_data)

        self.original_error_bar.setOpts(height=[original_rms])
        self.best_error_bar.setOpts(height=[best_rms])
        self.latest_error_bar.setOpts(height=[error_value])

    def process_events(self):
        """Keep the GUI responsive."""
        self.app.processEvents()