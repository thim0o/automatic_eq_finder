# src/automatic_eq_equalizer/ui/realtime_plotter.py

"""Real-time plotting utilities for Automatic EQ Equalizer with a dark theme."""

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore

# This makes the entire UI, including buttons and backgrounds, dark.
DARK_STYLESHEET = """
    QWidget {
        background-color: #1e1e1e;
        color: #dcdcdc;
        font-family: Segoe UI, sans-serif;
        font-size: 14pt;
    }
    QMainWindow {
        background-color: #1e1e1e;
    }
    QGroupBox {
        background-color: #2d2d2d;
        border: 1px solid #444444;
        border-radius: 5px;
        margin-top: 1ex; /* leave space at the top for the title */
        font-weight: bold;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top center; /* position at the top center */
        padding: 0 3px;
        background-color: #2d2d2d;
        color: #dcdcdc;
    }
    QCheckBox {
        color: #dcdcdc;
    }
    QCheckBox::indicator {
        width: 13px;
        height: 13px;
        border: 1px solid #555555;
        border-radius: 3px;
    }
    QCheckBox::indicator:checked {
        background-color: #55aaff; /* A pleasant blue for checked state */
        border: 1px solid #55aaff;
    }
"""


class RealTimePlotter:
    """
    Plot measured, baseline, and target responses plus error, with a dark theme UI.
    """

    def __init__(self, app):
        self.app = app
        # Apply the dark stylesheet to the entire application
        self.app.setStyleSheet(DARK_STYLESHEET)

        # Set pyqtgraph's global background to match the dark theme
        pg.setConfigOption("background", "#1e1e1e")
        pg.setConfigOption("foreground", "#dcdcdc")

        # Use a QMainWindow for a more standard application feel
        self.win = QtWidgets.QMainWindow()
        self.win.setWindowTitle("Real-time EQ Optimization")

        # Create a central widget and a main vertical layout
        central_widget = QtWidgets.QWidget()
        self.win.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)

        # --- Create Plotting Widget ---
        self.plot_widget = pg.GraphicsLayoutWidget()
        main_layout.addWidget(self.plot_widget)

        # --- Create Controls Widget ---
        controls_group = QtWidgets.QGroupBox("Toggle Visibility")
        controls_layout = QtWidgets.QHBoxLayout()
        controls_group.setLayout(controls_layout)
        controls_group.setMaximumHeight(80)
        main_layout.addWidget(controls_group)

        # --- Configure Plots with New Layout ---
        self.freq_plot = self.plot_widget.addPlot(
            title="Frequency Response", row=0, col=0, colspan=2
        )
        self.freq_plot.setLogMode(x=True, y=False)
        self.freq_plot.setLabels(left="Magnitude (dB)", bottom="Frequency (Hz)")
        self.freq_plot.showGrid(x=True, y=True, alpha=0.3)
        self.freq_plot.getAxis("bottom").setGrid(150)
        self.freq_plot.getAxis("left").setGrid(150)
        self.freq_plot.addLegend(
            labelTextSize="18pt", brush=pg.mkBrush(45, 45, 45, 150)
        )

        self.error_plot = self.plot_widget.addPlot(
            title="Optimization Error", row=1, col=0
        )
        self.error_plot.setLabels(left="Error (dB RMS)", bottom="Iteration")
        self.error_plot.showGrid(x=True, y=True, alpha=0.3)

        self.error_bar_plot = self.plot_widget.addPlot(
            title="Summary RMS Error", row=1, col=1
        )
        self.error_bar_plot.setLabels(left="RMS Error (dB)")
        self.error_bar_plot.showGrid(x=False, y=True, alpha=0.3)
        self.error_bar_plot.setXRange(-0.5, 2.5)
        self.error_bar_plot.getAxis("bottom").setTicks(
            [[(0, "Original"), (1, "Best"), (2, "Latest")]]
        )

        # --- Initialize Curves and Bars ---
        self.original_curve = self.freq_plot.plot(
            pen=pg.mkPen("#00ffff", width=1.5), name="Original"
        )  # Cyan
        self.best_corrected_curve = self.freq_plot.plot(
            pen=pg.mkPen("#ffff00", width=1.5), name="Best Corrected"
        )  # Yellow
        self.latest_corrected_curve = self.freq_plot.plot(
            pen=pg.mkPen("#ff00ff", width=1.5), name="Latest Corrected"
        )  # Magenta
        self.eq_profile_curve = self.freq_plot.plot(
            pen=pg.mkPen("w", width=2, style=QtCore.Qt.PenStyle.DashLine),
            name="EQ Profile",
        )
        self.target_curve = self.freq_plot.plot(
            pen=pg.mkPen("#00ff00", width=2.5, style=QtCore.Qt.PenStyle.SolidLine),
            name="Target Curve",
        )  # Bright Green

        # Use a brighter green for better visibility on dark background
        self.error_curve = self.error_plot.plot(
            pen=pg.mkPen("#00ff00", width=1.5), symbol="o", symbolBrush="#00ff00"
        )

        # Bar graph colors are already bright, so they are fine
        self.original_error_bar = pg.BarGraphItem(
            x=[0], height=[0], width=0.6, brush="#00ffff"
        )
        self.best_error_bar = pg.BarGraphItem(
            x=[1], height=[0], width=0.6, brush="#ffff00"
        )
        self.latest_error_bar = pg.BarGraphItem(
            x=[2], height=[0], width=0.6, brush="#ff00ff"
        )

        self.error_bar_plot.addItem(self.original_error_bar)
        self.error_bar_plot.addItem(self.best_error_bar)
        self.error_bar_plot.addItem(self.latest_error_bar)

        # --- Create and Connect Checkboxes ---
        checkboxes_defs = {
            "Original": "chk_original",
            "Best Corrected": "chk_best",
            "Latest Corrected": "chk_latest",
            "EQ Profile": "chk_eq_profile",
            "Target Curve": "chk_target",
            "Filter Visuals": "chk_filters",
        }
    
        for text, attr_name in checkboxes_defs.items():
            chk = QtWidgets.QCheckBox(text)
            chk.setChecked(True)
            controls_layout.addWidget(chk)
            setattr(self, attr_name, chk)  # Store checkbox as self.chk_original, etc.
    
        # Connect signals to slots
        self.chk_original.toggled.connect(self.original_curve.setVisible)
        self.chk_best.toggled.connect(self.best_corrected_curve.setVisible)
        self.chk_latest.toggled.connect(self.latest_corrected_curve.setVisible)
        self.chk_eq_profile.toggled.connect(self.eq_profile_curve.setVisible)
        self.chk_target.toggled.connect(self.target_curve.setVisible)
        self.chk_filters.toggled.connect(self.toggle_filter_visuals)
    
        # --- Start / Stop Buttons ---
        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        # Initially, the process is not running so Stop is disabled
        self.btn_stop.setEnabled(False)
        # Add a little spacing before buttons (stretch keeps checkboxes left-aligned)
        controls_layout.addStretch(1)
        controls_layout.addWidget(self.btn_start)
        controls_layout.addWidget(self.btn_stop)
    
        # --- Data Storage ---
        self.error_data = []
        self.iteration_data = []
        self.filter_visuals = []

        # --- Show Window ---
        self.win.showMaximized()

    def toggle_filter_visuals(self, checked):
        """Show or hide all plot items related to filter visualization."""
        for item in self.filter_visuals:
            item.setVisible(checked)

    def _update_filter_visuals(self, freqs, target_curve_data, filter_list):
        """Clear old filter visuals and draw new ones."""
        for item in self.filter_visuals:
            self.freq_plot.removeItem(item)
        self.filter_visuals.clear()

        if freqs is None or len(freqs) == 0:
            return

        is_visible = self.chk_filters.isChecked()

        for fc, gain, q in filter_list:
            if gain == 0:
                continue

            target_level_at_fc = np.interp(fc, freqs, target_curve_data)
            # Use a brighter orange for boost on dark background
            pen_color = pg.mkPen(
                "#ff0000" if gain < 0 else "#ffA500", width=1.5
            )  # Red for cut, Orange for boost

            gain_line = pg.PlotDataItem(
                x=[fc, fc],
                y=[target_level_at_fc, target_level_at_fc + gain],
                pen=pen_color,
            )
            bandwidth = fc / q
            q_line = pg.PlotDataItem(
                x=[fc - bandwidth / 2, fc + bandwidth / 2],
                y=[target_level_at_fc + gain, target_level_at_fc + gain],
                pen=pen_color,
            )

            gain_line.setVisible(is_visible)
            q_line.setVisible(is_visible)

            self.freq_plot.addItem(gain_line)
            self.freq_plot.addItem(q_line)
            self.filter_visuals.extend([gain_line, q_line])

    def update_freq_response(
        self,
        freqs,
        original,
        best_corrected,
        latest_corrected,
        eq_curve,
        target_curve_data,
        filter_list,
    ):
        """Update frequency response curves, target curve, and filter visuals."""
        self.original_curve.setData(freqs, original)
        self.best_corrected_curve.setData(freqs, best_corrected)
        self.latest_corrected_curve.setData(freqs, latest_corrected)
        self.eq_profile_curve.setData(freqs, eq_curve)
        self.target_curve.setData(freqs, target_curve_data)
        self._update_filter_visuals(freqs, target_curve_data, filter_list)

    def update_error(self, iteration, error_value, original_rms, best_rms):
        """Append an error value and update RMS error bars."""
        self.error_data.append(error_value)
        self.iteration_data.append(iteration)
        self.error_curve.setData(self.iteration_data, self.error_data)
        self.original_error_bar.setOpts(height=[original_rms])
        self.best_error_bar.setOpts(height=[best_rms])
        self.latest_error_bar.setOpts(height=[error_value])

    def process_events(self):
        """Keep the GUI responsive."""
        self.app.processEvents()
