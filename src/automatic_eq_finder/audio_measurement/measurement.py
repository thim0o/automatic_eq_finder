import numpy as np
import pyaudio
import matplotlib.pyplot as plt
from scipy import signal
import time
import threading
import queue
from scipy.io import wavfile
from scipy.stats import t
import csv
import os
import sys


class FrequencyResponseMeasurement:
    """
    Performs frequency response measurements using a log sweep and deconvolution.

    Attributes:
        sample_rate (int): Sampling rate for audio (default: 44100 Hz).
        chunk_size (int): Chunk size for audio processing (default: 4096 samples).
        start_freq (int): Start frequency of the sweep (default: 20 Hz).
        end_freq (int): End frequency of the sweep (default: 20000 Hz).
        sweep_duration (float): Duration of the frequency sweep in seconds (default: 3.0).
        num_averages (int): Number of measurements to average (default: 1).
        confidence_interval_width (float): Width of the confidence interval (default: 0.95).

    Methods:
        generate_log_sweep(): Generates a logarithmic frequency sweep signal.
        generate_inverse_filter(): Generates the inverse filter for deconvolution.
        play_audio(): Plays audio data through the speakers.
        record_audio(): Records audio data from the microphone.
        playback_thread(): Thread function for playing audio in the background.
        calculate_impulse_response(): Calculates the impulse response using deconvolution.
        calculate_frequency_response(): Calculates frequency response from impulse response.
        apply_smoothing(): Applies 1/n octave smoothing to the frequency response.
        run_measurement(): Runs the full frequency response measurement process.
        save_frequency_response_csv(): Saves frequency response data to a CSV file.
        plot_results(): Plots the frequency response curve with confidence intervals.
        cleanup(): Terminates PyAudio resources.
    """
    def __init__(self,
                 sample_rate=44100,
                 chunk_size=4096,
                 start_freq=20,
                 end_freq=20000,
                 sweep_duration=2.0,
                 num_averages=1,
                 confidence_interval_width=0.95):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.start_freq = start_freq
        self.end_freq = end_freq
        self.sweep_duration = sweep_duration
        self.num_averages = num_averages
        self.confidence_interval_width = confidence_interval_width

        self.p = pyaudio.PyAudio()
        self.audio_queue = queue.Queue()
        self.results = {}
        # cooperative stop event and playback thread reference
        self._stop_event = threading.Event()
        self._playback_thread_obj = None


    def generate_log_sweep(self):
        """Generate a logarithmic frequency sweep"""
        # Number of samples for the sweep
        num_samples = int(self.sweep_duration * self.sample_rate)

        # Time array
        t = np.linspace(0, self.sweep_duration, num_samples)

        # Logarithmic frequency sweep calculation
        # This equation makes the frequency increase exponentially with time
        w1 = 2 * np.pi * self.start_freq
        w2 = 2 * np.pi * self.end_freq
        K = (self.sweep_duration * w1) / np.log(w2 / w1)
        L = (self.sweep_duration) / np.log(w2 / w1)

        # Phase calculation for log sweep
        phase = K * (np.exp(t / L) - 1.0)

        # Generate the sweep
        sweep = 0.5 * np.sin(phase)

        # Apply a fade in/out to avoid clicks
        fade_samples = int(0.05 * self.sample_rate)  # 50ms fade
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)

        sweep[:fade_samples] *= fade_in
        sweep[-fade_samples:] *= fade_out

        return sweep.astype(np.float32)

    def generate_inverse_filter(self, sweep):
        """Generate the inverse filter for deconvolution"""
        # For log sweep, the inverse filter is the time-reversed sweep
        # with amplitude correction for 6dB/octave

        # Create time-reversed sweep
        inv_sweep = sweep[::-1]

        # Apply amplitude correction for pink spectrum
        n = len(inv_sweep)
        freq = np.fft.rfftfreq(n, 1 / self.sample_rate)
        sweep_fft = np.fft.rfft(inv_sweep)

        # Apply 6dB/octave amplitude correction (f^0.5 in frequency domain)
        amplitude_correction = np.ones_like(freq)
        mask = (freq >= self.start_freq) & (freq <= self.end_freq)
        amplitude_correction[mask] = np.sqrt(freq[mask] / self.start_freq)

        # Apply correction
        corrected_fft = sweep_fft * amplitude_correction

        # Back to time domain
        inverse_filter = np.fft.irfft(corrected_fft)

        # Normalize
        inverse_filter = inverse_filter / np.max(np.abs(inverse_filter))

        return inverse_filter.astype(np.float32)

    def play_audio(self, audio_data, data_format='float32'):
        """Play audio through speakers, now with format awareness"""
        if data_format == 'float32':
            format = pyaudio.paFloat32
        elif data_format == 'int16':
            format = pyaudio.paInt16
        else:
            raise ValueError(f"Unsupported data_format: {data_format}")

        stream = self.p.open(format=format,
                             channels=1,
                             rate=self.sample_rate,
                             output=True)
        stream.write(audio_data.tobytes())
        stream.stop_stream()
        stream.close()

    def record_audio(self, duration):
        """Record audio from microphone"""
        frames = []
        stream = self.p.open(format=pyaudio.paFloat32,
                             channels=1,
                             rate=self.sample_rate,
                             input=True,
                             frames_per_buffer=self.chunk_size)

        # Calculate total chunks to read
        total_chunks = int(self.sample_rate * duration / self.chunk_size)

        for _ in range(total_chunks + 1):  # +1 to ensure we get all audio
            data = stream.read(self.chunk_size)
            frames.append(data)

        stream.stop_stream()
        stream.close()

        # Convert frames to numpy array
        audio_data = np.frombuffer(b''.join(frames), dtype=np.float32)
        return audio_data

    def playback_thread(self):
        """Thread function for playing audio"""
        while True:
            item = self.audio_queue.get()
            if item is None:  # Signal to exit
                break
            audio_data, data_format = item
            self.play_audio(audio_data, data_format=data_format)
            self.audio_queue.task_done()

    def calculate_impulse_response(self, recorded_sweep, sweep, inverse_filter):
        """Calculate the impulse response using deconvolution"""
        # Ensure both arrays have the same length for FFT operations
        n = min(len(recorded_sweep), len(inverse_filter))

        # Truncate both arrays to the same length
        recorded_sweep = recorded_sweep[:n]
        inverse_filter = inverse_filter[:n]

        # Simple cross-correlation method
        ir = signal.fftconvolve(recorded_sweep, inverse_filter, mode='full')

        # Find peak of impulse response
        peak_idx = np.argmax(np.abs(ir))

        # Trim the impulse response around the peak
        ir_len = min(len(ir), self.sample_rate // 2)  # Use half a second
        start_idx = max(0, peak_idx - ir_len // 4)
        ir_trimmed = ir[start_idx:start_idx + ir_len]

        return ir_trimmed

    def calculate_frequency_response(self, impulse_response):
        """Calculate frequency response from impulse response"""
        # Apply window to reduce spectral leakage
        window = signal.windows.hann(len(impulse_response))
        windowed_ir = impulse_response * window

        # Calculate FFT
        n_fft = 2 ** int(np.ceil(np.log2(len(windowed_ir))))  # Next power of 2
        fr = np.fft.rfft(windowed_ir, n=n_fft)

        # Convert to magnitude in dB
        magnitude = 20 * np.log10(np.abs(fr) + 1e-10)

        # Get corresponding frequencies
        frequencies = np.fft.rfftfreq(n_fft, 1 / self.sample_rate)

        # Apply 1/3 octave smoothing
        magnitude_smoothed = self.apply_smoothing(frequencies, magnitude)

        return frequencies, magnitude_smoothed

    def apply_smoothing(self, frequencies, magnitude, fraction=3):
        """Apply 1/n octave smoothing to the frequency response"""
        # Skip smoothing if we have very few points
        if len(frequencies) < 10:
            return magnitude

        smoothed = np.zeros_like(magnitude)

        for i, f in enumerate(frequencies):
            if f == 0:  # Skip DC
                smoothed[i] = magnitude[i]
                continue

            # Calculate lower and upper frequency bounds for averaging
            f_low = f / (2 ** (0.5 / fraction))
            f_high = f * (2 ** (0.5 / fraction))

            # Find indices within this range
            indices = np.where((frequencies >= f_low) & (frequencies <= f_high))[0]

            if len(indices) > 0:
                # Calculate weighted average (closer frequencies get higher weight)
                weights = 1.0 / (1.0 + np.abs(np.log(frequencies[indices] / f)))
                smoothed[i] = np.sum(magnitude[indices] * weights) / np.sum(weights)
            else:
                smoothed[i] = magnitude[i]

        return smoothed

    def run_measurement(self):
        """Run the full frequency response measurement"""
        print("Starting frequency response measurement with rapid sweep...")
 
        # cooperative stop: clear any previous stop flag
        try:
            self._stop_event.clear()
        except Exception:
            pass
        # Start playback thread
        playback_thread = threading.Thread(target=self.playback_thread)
        playback_thread.daemon = True
        playback_thread.start()
        self._playback_thread_obj = playback_thread

        # Generate sweep signal
        sweep = self.generate_log_sweep()
        print(f"Sweep length: {len(sweep)} samples")

        # Generate inverse filter for deconvolution
        inverse_filter = self.generate_inverse_filter(sweep)
        print(f"Inverse filter length: {len(inverse_filter)} samples")

        # Add a bit of silence at the end to capture room decay
        silence = np.zeros(int(self.sample_rate * 0.5), dtype=np.float32)
        test_signal = np.concatenate([sweep, silence])

        # Save the sweep signal for reference
        wavfile.write('test_sweep.wav', self.sample_rate, sweep)

        # Storage for multiple measurements
        all_freqs = []
        all_magnitudes = []
        all_individual_magnitudes = []

        for i in range(self.num_averages):
            # cooperative stop check
            try:
                if getattr(self, "_stop_event", None) is not None and self._stop_event.is_set():
                    try:
                        # signal playback thread to finish if waiting on queue
                        self.audio_queue.put(None)
                        if self._playback_thread_obj is not None:
                            self._playback_thread_obj.join(timeout=0.5)
                    except Exception:
                        pass
                    return None
            except Exception:
                pass
            print(f"Running measurement {i + 1}/{self.num_averages}")

            # Queue the audio for playback - UNPROCESSED sweep is float32
            self.audio_queue.put((test_signal, 'float32'))

            # Short delay to ensure recording starts after playback
            time.sleep(0.05)

            # Record the response
            record_duration = len(test_signal) / self.sample_rate + 0.2  # Add margin
            recorded_data = self.record_audio(record_duration)
            print(f"Recorded data length: {len(recorded_data)} samples")

            # Save the recorded data for debugging
            if i == 0:
                wavfile.write('recorded_sweep.wav', self.sample_rate, recorded_data)

            # Ensure no length mismatch by trimming if necessary
            if len(recorded_data) > len(test_signal) * 2:
                recorded_data = recorded_data[:len(test_signal) + self.sample_rate]
                print(f"Trimmed recorded data to {len(recorded_data)} samples")

            # Calculate impulse response - pass sweep for alignment
            try:
                impulse_response = self.calculate_impulse_response(recorded_data, sweep, inverse_filter)
                print(f"Impulse response length: {len(impulse_response)} samples")

                # Save impulse response
                if i == 0:
                    normalized_ir = impulse_response / (np.max(np.abs(impulse_response)) + 1e-10)
                    wavfile.write('impulse_response.wav', self.sample_rate, normalized_ir)

                # Calculate frequency response
                frequencies, magnitude = self.calculate_frequency_response(impulse_response)

                # Store results
                all_freqs.append(frequencies)
                all_magnitudes.append(magnitude)

            except Exception as e:
                print(f"Error processing measurement {i + 1}: {e}")

            # Wait for playback to finish
            self.audio_queue.join()

        # Signal playback thread to exit
        self.audio_queue.put(None)
        playback_thread.join()

        # Combine results - we may have different frequency arrays, so we need to interpolate
        if len(all_magnitudes) > 0:
            # Use the first frequency array as reference
            freq_ref = all_freqs[0]

            # Interpolate all other measurements to match the reference frequencies
            interpolated_magnitudes = []

            for i in range(len(all_magnitudes)):
                if len(all_freqs[i]) > 10:  # Skip if we have too few points
                    interpolated = np.interp(
                        freq_ref,
                        all_freqs[i],
                        all_magnitudes[i],
                        left=all_magnitudes[i][0],
                        right=all_magnitudes[i][-1]
                    )
                    interpolated_magnitudes.append(interpolated)

            if interpolated_magnitudes: # Check if interpolated_magnitudes is not empty
                # Average the results
                avg_magnitude = np.mean(interpolated_magnitudes, axis=0)

                # Store only frequencies within our test range
                mask = (freq_ref >= self.start_freq) & (freq_ref <= self.end_freq)
                self.results = {
                    'frequencies': freq_ref[mask],
                    'magnitude': avg_magnitude[mask],
                    'individual_magnitudes': np.array(interpolated_magnitudes)[:, mask]
                }

                print("Measurement complete!")
                return self.results

        print("No valid measurements obtained.")
        return None

    def save_frequency_response_csv(self, filename='frequency_response.csv'):
        """Save frequency response data to a CSV file."""
        if not self.results or 'frequencies' not in self.results or 'magnitude' not in self.results:
            print("No frequency response data to save.")
            return

        frequencies = self.results['frequencies']
        magnitude = self.results['magnitude']

        with open(filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['frequency', 'raw'])  # Write header row
            for freq, mag in zip(frequencies, magnitude):
                csv_writer.writerow([f"{freq:.2f}", f"{mag:.2f}"]) # Format to two decimal places
        print(f"Frequency response data saved to '{filename}'")


    def plot_results(self):
        """Plot the frequency response curve with CI bands and auto-fitted y-axis"""
        if not self.results or 'frequencies' not in self.results:
            print("No results to plot. Run measurement first.")
            return

        frequencies = self.results['frequencies']
        magnitude = self.results['magnitude']
        individual_magnitudes = self.results['individual_magnitudes']

        if len(frequencies) < 10:
            print("Not enough data points to create a meaningful plot.")
            return

        # Normalize the magnitude so 1kHz is at 0dB (or closest available frequency)
        idx_1khz = np.argmin(np.abs(frequencies - 1000))
        magnitude = magnitude - magnitude[idx_1khz]
        individual_magnitudes = individual_magnitudes - individual_magnitudes[:, idx_1khz][:, np.newaxis]

        # Calculate confidence interval
        std_dev_magnitude = np.std(individual_magnitudes, axis=0)
        n = self.num_averages
        alpha = 1.0 - self.confidence_interval_width
        degrees_freedom = n - 1
        t_critical = t.ppf(1.0 - alpha / 2.0, degrees_freedom) if degrees_freedom > 0 else 1.96
        confidence_interval = t_critical * std_dev_magnitude / np.sqrt(n) if n > 1 else std_dev_magnitude

        upper_bound = magnitude + confidence_interval
        lower_bound = magnitude - confidence_interval

        plt.figure(figsize=(12, 8))
        #plt.semilogx(frequencies, magnitude, label='Average Response') # Can use semilogx if needed
        plt.plot(frequencies, magnitude, label='Average Response')
        plt.fill_between(frequencies, lower_bound, upper_bound, color='blue', alpha=0.2, label=f'{self.confidence_interval_width*100:.0f}% CI')
        plt.title('Speaker Frequency Response with Confidence Interval')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Relative Magnitude (dB)')
        plt.grid(True, which="both", ls="-")
        plt.xlim(self.start_freq, self.end_freq)
        plt.autoscale(axis='y')

        # Add vertical lines at common reference points
        for f in [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]:
            if self.start_freq <= f <= self.end_freq:
                plt.axvline(x=f, color='r', linestyle='--', alpha=0.3)
                plt.text(f, plt.ylim()[0] * 0.98, str(f), ha='center', fontsize=8)

        # Add horizontal lines at reference levels
        y_range = np.ptp(plt.ylim())
        step = 5 if y_range > 40 else 2 if y_range > 20 else 1
        start_level = int(plt.ylim()[0] // step) * step
        levels = np.arange(start_level, plt.ylim()[1], step)

        for level in levels:
            if level != 0:
                plt.axhline(y=level, color='g', linestyle='--', alpha=0.3)
                plt.text(self.start_freq * 1.1, level, f"{level} dB", va='center', fontsize=8)

        plt.axhline(y=0, color='k', linestyle='-', linewidth=0.8)
        plt.legend()
        plt.tight_layout()
        plt.savefig('frequency_response.png', dpi=150)
        plt.show()

    def cleanup(self):
        """Clean up resources"""
        self.p.terminate()
 
    def stop(self):
        """Request cooperative stop of an in-progress measurement."""
        try:
            self._stop_event.set()
        except Exception:
            pass
        try:
            # unblock playback thread if waiting on the queue
            self.audio_queue.put(None)
        except Exception:
            pass