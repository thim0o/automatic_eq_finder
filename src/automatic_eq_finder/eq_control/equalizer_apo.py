import os

class EqualizerPreset:
    """
    Represents an Equalizer APO preset configuration.

    Attributes:
        preamp (float): Preamp value in dB (default: 0.0).
        filters (list): List of filter dictionaries, each containing:
            - enabled (bool): True if the filter is enabled.
            - filter_code (str): Filter type code (e.g., 'LSC', 'PK', 'HSC').
            - fc (float): Center frequency in Hz.
            - gain (float): Gain in dB.
            - q (float): Q factor.

    Methods:
        set_preamp(preamp: float): Sets the preamp value.
        add_filter(enabled: bool, filter_code: str, fc: float, gain: float, q: float): Adds a filter to the preset.
        remove_filter(index: int): Removes the filter at the given index.
        to_string() -> str: Converts the preset to a string formatted for Equalizer APO.
        apply_to_file(config_path: str): Writes the preset configuration to a file.
        load_from_file(file_path: str) -> EqualizerPreset: Loads a preset from a file.
        reset_preset(): Creates and returns a reset preset (class method).

    Example:
        preset = EqualizerPreset(preamp=-3.0)
        preset.add_filter(True, "PK", 100, 6.0, 1.0)
        preset.apply_to_file("config.txt")
    """
    def __init__(self, preamp: float = 0.0):
        """
        Initialize a preset with a given preamp value and an empty list of filters.
        """
        self.preamp = preamp
        self.filters = []  # List to store filter dictionaries

    def set_preamp(self, preamp: float):
        """
        Set the preamp value.
        """
        self.preamp = preamp

    def add_filter(self, enabled: bool, filter_code: str, fc: float, gain: float, q: float):
        """
        Add a filter to the preset.

        :param enabled: True if the filter is enabled; False otherwise.
        :param filter_code: Filter type (e.g., 'LSC', 'PK', 'HSC').
        :param fc: Center frequency in Hz.
        :param gain: Gain in dB.
        :param q: Q factor.
        """
        filter_info = {
            "enabled": enabled,
            "filter_code": filter_code,
            "fc": fc,
            "gain": gain,
            "q": q
        }
        self.filters.append(filter_info)

    def remove_filter(self, index: int):
        """
        Remove the filter at the given index (0-indexed).
        """
        try:
            del self.filters[index]
        except IndexError:
            raise ValueError("Filter index out of range.")

    def to_string(self) -> str:
        """
        Convert the current preset into a text string formatted for Equalizer APO.
        """
        lines = []
        # Preamp line
        lines.append(f"Preamp: {self.preamp:.2f} dB")
        # Filter lines (numbering filters starting at 1)
        for i, filt in enumerate(self.filters, start=1):
            status = "ON" if filt["enabled"] else "OFF"
            line = (
                f"Filter {i}: {status} {filt['filter_code']} Fc {filt['fc']:.1f} Hz "
                f"Gain {filt['gain']:.1f} dB Q {filt['q']:.2f}"
            )
            lines.append(line)
        return "\n".join(lines)

    def apply_to_file(self, config_path: str):
        """
        Write the preset configuration to the given file path.
        """
        try:
            with open(config_path, "w") as f:
                f.write(self.to_string())
            print(f"Preset applied successfully to {config_path}")
        except Exception as e:
            print(f"Failed to write preset to {config_path}: {e}")
            raise

    @classmethod
    def load_from_file(cls, file_path: str):
        """
        Load a preset configuration from a file and return an EqualizerPreset object.
        Assumes the first line is the preamp setting and subsequent lines are filter definitions.
        """
        try:
            with open(file_path, "r") as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading preset file {file_path}: {e}")
            raise

        lines = content.splitlines()
        if not lines:
            raise ValueError("Preset file is empty.")

        # Parse the preamp value from the first line (e.g., "Preamp: -5.50 dB")
        preamp_line = lines[0]
        if not preamp_line.startswith("Preamp:"):
            raise ValueError("Invalid preset file: missing preamp line.")
        try:
            preamp_value = float(preamp_line.split("Preamp:")[1].strip().split()[0])
        except Exception as e:
            raise ValueError("Invalid preamp value in preset file.") from e

        preset = cls(preamp=preamp_value)
        # Parse each filter line
        for line in lines[1:]:
            try:
                # Expected format:
                # Filter 1: ON LSC Fc 105.0 Hz Gain -1.3 dB Q 0.70
                parts = line.split()
                # parts[2] is status, parts[3] is filter code.
                enabled = True if parts[2].upper() == "ON" else False
                filter_code = parts[3]
                # parts[5] is the frequency value after 'Fc'
                fc = float(parts[5])
                # parts[8] is the gain value after 'Gain'
                gain = float(parts[8])
                # parts[11] is the Q value after 'Q'
                q = float(parts[11])
                preset.add_filter(enabled, filter_code, fc, gain, q)
            except Exception as e:
                print(f"Error parsing line: '{line}'. Error: {e}")
        return preset

    @classmethod
    def reset_preset(cls):
        """
        Creates and returns a preset object with no filters and 0 dB preamp,
        effectively resetting the EQ.
        """
        return cls(preamp=0.0)