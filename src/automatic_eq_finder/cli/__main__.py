# src/automatic_eq_equalizer/cli/__main__.py

"""
Main entry point for launching the Automatic EQ Equalizer graphical user interface.
"""

from PyQt5 import QtWidgets
from automatic_eq_finder.ui.realtime_plotter import RealTimePlotter
from automatic_eq_finder.core.auto_eq import AutoEQIterative


def main():
    """Launch the Automatic EQ Equalizer UI."""
    print("Hello, Automatic EQ Equalizer!")

    # Set up the Qt Application
    app = QtWidgets.QApplication([])

    # Create the UI plotter and the worker thread
    plotter = RealTimePlotter(app)
    auto_eq = AutoEQIterative()

    # Connect the worker's signals to the plotter's slots
    auto_eq.update_freq_signal.connect(plotter.update_freq_response)
    auto_eq.update_error_signal.connect(plotter.update_error)

    # Start the worker thread
    auto_eq.start()

    # Execute the application
    app.exec_()


if __name__ == "__main__":
    main()