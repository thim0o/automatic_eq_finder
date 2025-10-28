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

    def start_worker(single_sweep=False):
        nonlocal auto_eq
        # If a worker is already running, ignore
        if auto_eq is not None and getattr(auto_eq, "isRunning", lambda: False)():
            print("AutoEQ worker already running.")
            return
        
        # Create and wire a new worker thread
        auto_eq = AutoEQIterative(
            single_sweep=single_sweep,
            bass_gain_db=plotter.bass_gain_db,
            corner_freq=plotter.corner_freq,
            tilt_db_per_decade=plotter.tilt_db_per_decade
        )
        # Connect signals to update the UI
        auto_eq.update_freq_signal.connect(plotter.update_freq_response)
        auto_eq.update_error_signal.connect(plotter.update_error)
        # Connect microphone/device name info so the UI shows which mic is used
        try:
            auto_eq.mic_info_signal.connect(plotter.set_mic_name)
        except Exception:
            # If the attribute is not present for some reason, ignore silently
            pass
        
        # Connect finished signal for cleanup, especially important for single sweeps
        auto_eq.finished.connect(worker_finished)
        
        auto_eq.start()
        
        # Only disable buttons if it's a continuous run
        if not single_sweep:
            plotter.btn_start.setEnabled(False)
            plotter.btn_stop.setEnabled(True)
        else:
            # Disable start/stop buttons temporarily during the initial sweep
            plotter.btn_start.setEnabled(False)
            plotter.btn_stop.setEnabled(False)
            
        print(f"AutoEQ worker started (Single Sweep: {single_sweep}).")

    def worker_finished():
        nonlocal auto_eq
        print("AutoEQ worker finished.")
        # Clean up the thread object
        if auto_eq is not None:
            auto_eq.deleteLater()
            auto_eq = None
        
        # Re-enable buttons
        plotter.btn_start.setEnabled(True)
        plotter.btn_stop.setEnabled(False)

    def stop_worker():
        nonlocal auto_eq
        if auto_eq is None:
            print("No worker to stop.")
            return
        # Request a graceful stop; the worker will check the flag and exit
        auto_eq.stop()
        # The worker_finished slot will handle re-enabling buttons when the thread exits
        print("Stop requested for AutoEQ worker.")

    # Wire UI buttons to the start/stop handlers
    plotter.btn_start.clicked.connect(lambda: start_worker(single_sweep=False))
    plotter.btn_stop.clicked.connect(stop_worker)
    
    # --- Trigger initial single sweep measurement ---
    start_worker(single_sweep=True)

    # Execute the application
    app.exec_()


if __name__ == "__main__":
    main()