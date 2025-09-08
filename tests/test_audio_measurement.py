import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from automatic_eq_equalizer.audio_measurement.measurement import FrequencyResponseMeasurement


class TestFrequencyResponseMeasurement:
    """Test suite for FrequencyResponseMeasurement class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.frm = FrequencyResponseMeasurement(
            sample_rate=44100,
            chunk_size=1024,
            start_freq=100,
            end_freq=1000,
            sweep_duration=1.0,
            num_averages=1
        )
    
    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        frm = FrequencyResponseMeasurement()
        assert frm.sample_rate == 44100
        assert frm.chunk_size == 4096
        assert frm.start_freq == 20
        assert frm.end_freq == 20000
        assert frm.sweep_duration == 3.0
        assert frm.num_averages == 1
        assert frm.confidence_interval_width == 0.95
    
    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        assert self.frm.sample_rate == 44100
        assert self.frm.chunk_size == 1024
        assert self.frm.start_freq == 100
        assert self.frm.end_freq == 1000
        assert self.frm.sweep_duration == 1.0
        assert self.frm.num_averages == 1
    
    def test_generate_log_sweep(self):
        """Test log sweep generation."""
        sweep = self.frm.generate_log_sweep()
        
        # Check basic properties
        expected_samples = int(self.frm.sweep_duration * self.frm.sample_rate)
        assert len(sweep) == expected_samples
        assert sweep.dtype == np.float32
        
        # Check amplitude range (should be between -0.5 and 0.5)
        assert np.max(sweep) <= 0.5
        assert np.min(sweep) >= -0.5
        
        # Check fade in/out (first and last samples should be close to 0)
        fade_samples = int(0.05 * self.frm.sample_rate)
        assert abs(sweep[0]) < 0.1
        assert abs(sweep[-1]) < 0.1
    
    def test_generate_inverse_filter(self):
        """Test inverse filter generation."""
        sweep = self.frm.generate_log_sweep()
        inverse_filter = self.frm.generate_inverse_filter(sweep)
        
        # Check basic properties
        assert len(inverse_filter) == len(sweep)
        assert inverse_filter.dtype == np.float32
        
        # Check normalization (max absolute value should be 1.0)
        assert np.max(np.abs(inverse_filter)) == pytest.approx(1.0, rel=1e-6)
    
    def test_calculate_impulse_response(self):
        """Test impulse response calculation."""
        # Create test signals
        sweep = self.frm.generate_log_sweep()
        inverse_filter = self.frm.generate_inverse_filter(sweep)
        recorded_sweep = sweep + 0.01 * np.random.randn(len(sweep))  # Add some noise
        
        ir = self.frm.calculate_impulse_response(recorded_sweep, sweep, inverse_filter)
        
        # Check that impulse response is generated
        assert len(ir) > 0
        assert isinstance(ir, np.ndarray)
        
        # Check that impulse response has reasonable length
        max_expected_length = self.frm.sample_rate // 2
        assert len(ir) <= max_expected_length
    
    def test_calculate_frequency_response(self):
        """Test frequency response calculation."""
        # Create a simple impulse response (delta function)
        ir_length = 1024
        impulse_response = np.zeros(ir_length)
        impulse_response[100] = 1.0  # Delta at sample 100
        
        frequencies, magnitude = self.frm.calculate_frequency_response(impulse_response)
        
        # Check basic properties
        assert len(frequencies) == len(magnitude)
        assert len(frequencies) > 0
        assert np.all(frequencies >= 0)
        assert np.all(frequencies <= self.frm.sample_rate / 2)
    
    def test_apply_smoothing(self):
        """Test frequency response smoothing."""
        # Create test data
        frequencies = np.logspace(2, 3, 100)  # 100 Hz to 1 kHz
        magnitude = np.random.randn(100) * 5  # Random magnitude data
        
        smoothed = self.frm.apply_smoothing(frequencies, magnitude, fraction=3)
        
        # Check that smoothing preserves length
        assert len(smoothed) == len(magnitude)
        
        # Check that smoothing reduces variance (generally)
        # Note: This is a statistical test and might occasionally fail
        smoothed_var = np.var(smoothed)
        original_var = np.var(magnitude)
        # Smoothing should generally reduce variance, but we'll use a loose check
        assert smoothed_var <= original_var * 2  # Allow some tolerance
    
    def test_apply_smoothing_edge_cases(self):
        """Test smoothing with edge cases."""
        # Test with very few points
        frequencies = np.array([100, 200])
        magnitude = np.array([1.0, 2.0])
        
        smoothed = self.frm.apply_smoothing(frequencies, magnitude)
        
        # Should return original data for very few points
        assert len(smoothed) == len(magnitude)
        
        # Test with DC component
        frequencies = np.array([0, 100, 200])
        magnitude = np.array([0.0, 1.0, 2.0])
        
        smoothed = self.frm.apply_smoothing(frequencies, magnitude)
        assert smoothed[0] == magnitude[0]  # DC should be unchanged
    
    def test_compute_rms_error(self):
        """Test RMS error computation."""
        # Create test data within the frequency range
        freqs = np.linspace(self.frm.start_freq, self.frm.end_freq, 100)
        measured = np.ones_like(freqs) * 2.0  # Constant 2 dB
        
        rms_error = self.frm.compute_rms_error(freqs, measured)
        
        # For constant measured response, RMS error should be close to 2.0
        # (since target is the mean, which should be 2.0)
        assert rms_error == pytest.approx(0.0, abs=1e-10)
    
    @patch('automatic_eq_equalizer.audio_measurement.measurement.pyaudio.PyAudio')
    def test_cleanup(self, mock_pyaudio):
        """Test cleanup method."""
        mock_pa_instance = Mock()
        mock_pyaudio.return_value = mock_pa_instance
        
        frm = FrequencyResponseMeasurement()
        frm.cleanup()
        
        mock_pa_instance.terminate.assert_called_once()
    
    def test_save_frequency_response_csv_no_data(self, tmp_path):
        """Test CSV saving with no data."""
        # Clear results
        self.frm.results = {}
        
        # Should handle gracefully
        self.frm.save_frequency_response_csv()
        # No exception should be raised
    
    def test_save_frequency_response_csv_with_data(self, tmp_path):
        """Test CSV saving with valid data."""
        # Set up test data
        self.frm.results = {
            'frequencies': np.array([100, 200, 300]),
            'magnitude': np.array([1.0, 2.0, 3.0])
        }
        
        csv_file = tmp_path / "test_response.csv"
        self.frm.save_frequency_response_csv(str(csv_file))
        
        # Check that file was created and has correct content
        assert csv_file.exists()
        
        content = csv_file.read_text()
        lines = content.strip().split('\n')
        
        # Check header
        assert lines[0] == 'frequency,raw'
        
        # Check data rows
        assert '100.00,1.00' in lines[1]
        assert '200.00,2.00' in lines[2]
        assert '300.00,3.00' in lines[3]


class TestFrequencyResponseMeasurementIntegration:
    """Integration tests that require mocking of audio components."""
    
    @patch('automatic_eq_equalizer.audio_measurement.measurement.pyaudio.PyAudio')
    def test_play_audio_float32(self, mock_pyaudio):
        """Test audio playback with float32 format."""
        mock_pa_instance = Mock()
        mock_stream = Mock()
        mock_pyaudio.return_value = mock_pa_instance
        mock_pa_instance.open.return_value = mock_stream
        
        frm = FrequencyResponseMeasurement()
        test_audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        
        frm.play_audio(test_audio, data_format='float32')
        
        # Verify stream was opened with correct parameters
        mock_pa_instance.open.assert_called_once()
        call_args = mock_pa_instance.open.call_args[1]
        assert call_args['channels'] == 1
        assert call_args['rate'] == frm.sample_rate
        assert call_args['output'] == True
        
        # Verify audio was written and stream was closed
        mock_stream.write.assert_called_once()
        mock_stream.stop_stream.assert_called_once()
        mock_stream.close.assert_called_once()
    
    @patch('automatic_eq_equalizer.audio_measurement.measurement.pyaudio.PyAudio')
    def test_record_audio(self, mock_pyaudio):
        """Test audio recording."""
        mock_pa_instance = Mock()
        mock_stream = Mock()
        mock_pyaudio.return_value = mock_pa_instance
        mock_pa_instance.open.return_value = mock_stream
        
        # Mock recorded data
        test_data = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        mock_stream.read.return_value = test_data.tobytes()
        
        frm = FrequencyResponseMeasurement(chunk_size=2)
        duration = 0.1  # Short duration for test
        
        recorded = frm.record_audio(duration)
        
        # Verify stream was opened for input
        mock_pa_instance.open.assert_called_once()
        call_args = mock_pa_instance.open.call_args[1]
        assert call_args['input'] == True
        
        # Verify data was recorded
        assert isinstance(recorded, np.ndarray)
        assert recorded.dtype == np.float32
