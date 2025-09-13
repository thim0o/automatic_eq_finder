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

        # Error over iteration plot
        self.error_plot = self.win.addPlot(title="Optimization Error", row=1, col=0)
        self.error_plot.setLabels(left="Error (dB RMS)", bottom="Iteration")
        self.error_plot.showGrid(x=True, y=True)

        # Initialize curves
        self.original_curve = self.freq_plot.plot(pen='c', name="Original")  # Cyan
        self.best_corrected_curve = self.freq_plot.plot(pen='y', name="Best Corrected")  # Yellow
        self.latest_corrected_curve = self.freq_plot.plot(pen='m', name="Latest Corrected")  # Magenta
        self.target_line = self.freq_plot.plot(
            pen=pg.mkPen((192, 192, 192), width=1, style=pg.QtCore.Qt.PenStyle.DashLine),
            name="Target",
        )
        self.error_curve = self.error_plot.plot(pen='g')  # Green

        # Error bar plot
        self.error_bar_plot = self.win.addPlot(title="Summary RMS Error", row=2, col=0)
        self.error_bar_plot.setLabels(left="RMS Error (dB)")
        self.error_bar_plot.showGrid(x=False, y=True)
        self.error_bar_plot.setXRange(-0.5, 2.5)  # For 3 bars at 0, 1, 2
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
        """Update frequency response curves and target line."""
        if target_value is None:
            target_value = 0
        self.original_curve.setData(freqs, original)
        self.best_corrected_curve.setData(freqs, best_corrected)
        self.latest_corrected_curve.setData(freqs, latest_corrected)
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
