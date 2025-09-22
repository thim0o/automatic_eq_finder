# tests/test_audio_measurement.py

import pytest
import numpy as np
from unittest.mock import Mock, patch

from automatic_eq_finder.audio_measurement.measurement import (
    FrequencyResponseMeasurement,
)

# This test file focuses on the individual components of the measurement module.
# These are "unit tests".


class TestFrequencyResponseMeasurement:
    """Test suite for the FrequencyResponseMeasurement class."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up a fresh instance for each test method."""
        self.frm = FrequencyResponseMeasurement(
            sample_rate=44100,
            sweep_duration=1.0,
        )

    def test_generate_log_sweep(self):
        """Tests that the generated sweep has the correct properties."""
        sweep = self.frm.generate_log_sweep()
        expected_samples = int(self.frm.sweep_duration * self.frm.sample_rate)

        assert len(sweep) == expected_samples
        assert sweep.dtype == np.float32
        assert np.max(np.abs(sweep)) <= 0.5

    def test_generate_inverse_filter(self):
        """Tests that the inverse filter is correctly generated and normalized."""
        sweep = self.frm.generate_log_sweep()
        inverse_filter = self.frm.generate_inverse_filter(sweep)

        assert len(inverse_filter) == len(sweep)
        assert inverse_filter.dtype == np.float32
        assert np.max(np.abs(inverse_filter)) == pytest.approx(1.0)

    def test_calculate_frequency_response(self):
        """Tests the FFT and smoothing logic on a simple impulse."""
        ir_length = 2048
        impulse_response = np.zeros(ir_length)
        impulse_response[100] = 1.0  # A perfect impulse

        frequencies, magnitude = self.frm.calculate_frequency_response(impulse_response)

        assert len(frequencies) == len(magnitude)
        assert len(frequencies) > 0
        assert np.all(frequencies >= 0)
        # A flat frequency response is expected from a perfect impulse
        assert np.std(magnitude) < 1.0

    # We use mocking here to test PyAudio interactions without needing real hardware.
    @patch("automatic_eq_finder.audio_measurement.measurement.pyaudio.PyAudio")
    def test_play_audio_mocked(self, mock_pyaudio_class):
        """Tests that play_audio opens and uses a PyAudio stream correctly."""
        mock_pa_instance = Mock()
        mock_stream = Mock()
        mock_pyaudio_class.return_value = mock_pa_instance
        mock_pa_instance.open.return_value = mock_stream

        # We need a new FRM instance to get the mocked PyAudio object
        frm_mocked = FrequencyResponseMeasurement()
        test_audio = np.zeros(1024, dtype=np.float32)

        frm_mocked.play_audio(test_audio, data_format="float32")

        mock_pa_instance.open.assert_called_once()
        mock_stream.write.assert_called_once()
        mock_stream.stop_stream.assert_called_once()
        mock_stream.close.assert_called_once()

    @patch("automatic_eq_finder.audio_measurement.measurement.pyaudio.PyAudio")
    def test_record_audio_mocked(self, mock_pyaudio_class):
        """Tests that record_audio opens and reads from an input stream."""
        mock_pa_instance = Mock()
        mock_stream = Mock()
        mock_pyaudio_class.return_value = mock_pa_instance
        mock_pa_instance.open.return_value = mock_stream

        # Simulate some audio data being read from the stream
        fake_audio_chunk = np.zeros(1024, dtype=np.float32).tobytes()
        mock_stream.read.return_value = fake_audio_chunk

        frm_mocked = FrequencyResponseMeasurement()
        recorded_data = frm_mocked.record_audio(duration=0.1)

        mock_pa_instance.open.assert_called_once()
        mock_stream.read.assert_called()
        assert isinstance(recorded_data, np.ndarray)
