# src/automatic_eq_finder/cli/__main__.py

"""
Main entry point for launching the Automatic EQ Equalizer graphical user interface.
"""

from PyQt5 import QtWidgets
from automatic_eq_finder.ui.realtime_plotter import RealTimePlotter
from automatic_eq_finder.core.auto_eq import AutoEQIterative


def main():
    """Launch the Automatic EQ Equalizer UI with Start/Stop controls."""
    print("Launching Automatic EQ Equalizer UI...")

    # Set up the Qt Application
    app = QtWidgets.QApplication([])

    # Create the UI plotter (worker is created on demand)
    plotter = RealTimePlotter(app)
    auto_eq = None  # Will hold the current worker instance

    def start_worker():
        nonlocal auto_eq
        # If a worker is already running, ignore
        if auto_eq is not None and getattr(auto_eq, "isRunning", lambda: False)():
            print("AutoEQ worker already running.")
            return
        # Create and wire a new worker thread
        auto_eq = AutoEQIterative()
        auto_eq.update_freq_signal.connect(plotter.update_freq_response)
        auto_eq.update_error_signal.connect(plotter.update_error)
        auto_eq.start()
        plotter.btn_start.setEnabled(False)
        plotter.btn_stop.setEnabled(True)
        print("AutoEQ worker started.")

    def stop_worker():
        nonlocal auto_eq
        if auto_eq is None:
            print("No worker to stop.")
            return
        # Request a graceful stop; the worker will check the flag and exit
        auto_eq.stop()
        plotter.btn_stop.setEnabled(False)
        plotter.btn_start.setEnabled(True)
        print("Stop requested for AutoEQ worker.")

    # Wire UI buttons to the start/stop handlers
    plotter.btn_start.clicked.connect(start_worker)
    plotter.btn_stop.clicked.connect(stop_worker)

    # Execute the application
    app.exec_()


if __name__ == "__main__":
    main()