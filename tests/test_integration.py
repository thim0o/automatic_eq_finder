# tests/test_integration.py

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from automatic_eq_finder.core.auto_eq import AutoEQIterative
from automatic_eq_finder.utils import generate_harman_target


@pytest.fixture
def fake_measurement_data():
    """
    Generates a realistic, predictable frequency response for testing.

    This simulates a speaker that already follows the ideal Harman target curve
    but has a single, sharp +15 dB peak at 100 Hz. This provides a clear,
    isolated problem for the optimization algorithm to solve.
    """
    freqs = np.linspace(20, 2000, 500)

    # 1. Generate the ideal target curve to use as a baseline.
    magnitude = generate_harman_target(freqs)

    # 2. Add a single, obvious flaw for the optimizer to find and correct.
    peak_freq_index = np.argmin(np.abs(freqs - 100))
    magnitude[peak_freq_index] += 15.0

    sample_rate = 44100
    return freqs, magnitude, sample_rate


def test_auto_eq_full_run_simulation(fake_measurement_data):
    """
    An integration test for the entire AutoEQIterative workflow.

    This test simulates a full run of the core logic by:
    - Mocking external dependencies (audio measurement, file system access).
    - Providing a predictable, simulated frequency response.
    - Verifying that the optimization algorithm correctly identifies the
      simulated audio flaw and generates a logical corrective filter.
    - Confirming that UI signals are emitted as expected.
    """
    # Patch external interactions to isolate the core logic for the test.
    with patch(
        "automatic_eq_finder.core.auto_eq.AutoEQIterative.measure_response"
    ) as mock_measure, patch(
        "automatic_eq_finder.core.auto_eq.AutoEQIterative.apply_filters"
    ) as mock_apply, patch(
        "time.sleep"
    ):

        # Configure the mocked measurement to return our predictable test data.
        mock_measure.return_value = fake_measurement_data

        # Initialize the worker thread that contains the core application logic.
        worker = AutoEQIterative()

        # Create mock "slots" to verify that the UI signals are emitted.
        mock_freq_slot = MagicMock()
        mock_error_slot = MagicMock()
        worker.update_freq_signal.connect(mock_freq_slot)
        worker.update_error_signal.connect(mock_error_slot)

        # Execute the main run method.
        worker.run()

        # --- VERIFICATION ---

        # 1. Verify that the essential I/O methods were called.
        mock_measure.assert_called()
        mock_apply.assert_called()

        # 2. Verify that the optimizer produced a list of filters.
        assert worker.best_filter_list, "The optimizer should have generated filters."

        # 3. Find the most significant filter designed to correct our 100 Hz peak.
        #    This is more robust than checking the first filter in a sorted list.
        target_filter = None
        max_abs_gain = 0
        for fc, gain, q in worker.best_filter_list:
            if 90 < fc < 110:
                if abs(gain) > max_abs_gain:
                    max_abs_gain = abs(gain)
                    target_filter = (fc, gain, q)

        # 4. Assert that a logical, corrective filter was actually found.
        assert (
            target_filter is not None
        ), "Optimizer failed to create a filter for the 100 Hz peak."

        fc, gain, q = target_filter

        # 5. Assert that the filter's properties make sense for the problem.
        assert (
            gain < -5
        ), "The filter's gain should be strongly negative to correct the +15dB peak."
        assert q > 0, "The Q factor must be a positive value."

        # 6. Verify that the worker emitted signals to update the user interface.
        mock_freq_slot.assert_called()
        mock_error_slot.assert_called()
