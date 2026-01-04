import numpy as np

class QualityAnalyzer:
    """
    Analyzes the quality of audio measurements to detect background noise and signal integrity.
    """
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate

    def calculate_pnr(self, impulse_response):
        """
        Calculates the Peak-to-Noise Ratio (PNR) of the impulse response.
        Higher is better.
        """
        if len(impulse_response) == 0:
            return 0.0

        # Find the peak of the impulse response
        peak_idx = np.argmax(np.abs(impulse_response))
        peak_val = np.abs(impulse_response[peak_idx])

        # Look at noise before the peak (pre-arrival)
        # We take a window of 50ms before the peak, or whatever is available
        pre_arrival_samples = int(0.05 * self.sample_rate)
        noise_start = max(0, peak_idx - pre_arrival_samples)
        noise_end = max(0, peak_idx - int(0.005 * self.sample_rate)) # 5ms before peak to avoid the rise

        if noise_end > noise_start + 10:
            noise_floor = impulse_response[noise_start:noise_end]
        else:
            # Fallback: use the very end of the IR if pre-arrival is too short
            noise_floor = impulse_response[-int(0.1 * self.sample_rate):]

        rms_noise = np.sqrt(np.mean(noise_floor**2)) + 1e-10
        pnr = 20 * np.log10(peak_val / rms_noise)
        
        return float(pnr)

    def estimate_noise_floor(self, recorded_data, tail_duration=0.4):
        """
        Estimates the ambient noise floor from the silence tail of the recording.
        Lower is better. Returns RMS value in dB (relative to full scale 1.0).
        """
        num_samples = int(tail_duration * self.sample_rate)
        if len(recorded_data) < num_samples:
            noise_segment = recorded_data
        else:
            noise_segment = recorded_data[-num_samples:]
            
        rms_noise = np.sqrt(np.mean(noise_segment**2)) + 1e-10
        rms_db = 20 * np.log10(rms_noise)
        
        return float(rms_db)

    def get_quality_score(self, pnr, noise_floor_db):
        """
        Combines metrics into a single quality score (0-100).
        PNR > 40dB is excellent, < 20dB is poor.
        Noise floor < -60dB is excellent, > -30dB is poor.
        """
        # Normalize PNR (20 to 50 dB range)
        pnr_score = np.clip((pnr - 20) / (50 - 20), 0, 1) * 100
        
        # Normalize Noise Floor (-30 to -70 dB range)
        # Note: noise_floor_db is negative, so -70 is better than -30
        noise_score = np.clip((noise_floor_db - (-30)) / (-70 - (-30)), 0, 1) * 100
        
        # Weighted average (PNR is usually more indicative of sweep quality)
        combined_score = (pnr_score * 0.7) + (noise_score * 0.3)
        
        return float(combined_score)

    def get_quality_color(self, score):
        """Returns a hex color code based on the quality score."""
        if score >= 95:
            return "#00FF00" # Green
        elif score >= 90:
            return "#FFFF00" # Yellow
        elif score >= 80:
            return "#FFA500" # Orange
        else:
            return "#FF0000" # Red
