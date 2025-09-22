# Automatic EQ Finder

<div align="center">

![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)![License](https://img.shields.io/badge/license-MIT-green.svg)![Status](https://img.shields.io/badge/status-active-brightgreen)![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)

</div>

This project is an intelligent, closed-loop system that automatically corrects your room's acoustics. It listens to your speakers, identifies frequency imbalances caused by room reflections, and generates a precise parametric EQ to deliver a balanced, high-fidelity listening experience.

From real-time audio measurement to the direct application of filter settings, this is an end-to-end solution for automated sound calibration.
<img width="2560" height="1392" alt="Screenshot 2025-09-13 211513" src="https://github.com/user-attachments/assets/202bd17f-90d7-45fe-a200-cbf5caf5fb43" />

*The application's real-time visualization dashboard after a successful optimization.*

---

## Project Overview

In any listening environment, sound is distorted by the room itself. Reflections from walls, ceilings, and furniture create unwanted peaks and dips in the frequency response, coloring the audio and masking details. This tool replaces the complex, manual process of acoustic calibration with a fully automated, data-driven system to achieve **a more balanced sound**.



https://github.com/user-attachments/assets/2f10da14-15c8-4784-9bd9-f1a999569794

*A video showing the real-time optimization process.*




---

## Core Features

*   **Automated Acoustic Measurement:** Uses a logarithmic sine sweep to perform high-resolution frequency response measurements from a connected microphone.
*   **Hybrid Optimization Strategy:** A two-phase process ensures a robust and accurate EQ solution:
    1.  **Sequential Optimization:** A fast "greedy" algorithm builds an initial high-quality EQ by modeling the system and sequentially adding filters to correct the largest remaining errors.
    2.  **Iterative Fine-Tuning:** A closed-loop refinement stage repeatedly measures the corrected response, making small, data-driven adjustments to find the optimal filter parameters.
*   **Real-Time Visualization:** A multithreaded GUI, built with **PyQt5** and **pyqtgraph**, provides a responsive user interface and real-time plotting of all relevant acoustic data without interrupting the measurement process.
*   **Configurable Target Curve:** The system optimizes towards a target curve based on established acoustic principles (Harman-like), which is fully customizable in the project's configuration file.
*   **Direct System-Wide Integration:** Automatically generates and applies the final filter set to the **Equalizer APO** configuration file, enabling immediate, system-wide audio correction.

---

## System Architecture & Methodology

The application operates as a structured closed-loop control system to find the optimal EQ settings.

1.  **Initial Measurement:** The system captures the room's raw, uncorrected acoustic signature.
2.  **Analysis & Initial Correction:** This raw measurement is smoothed using a fractional-octave filter. This critical step ensures the optimization focuses on broad tonal imbalances relevant to human hearing, rather than narrow, insignificant measurement artifacts. The sequential optimizer then designs a full set of parametric filters based on this smoothed data.
3.  **Iterative Refinement:** The system enters a fine-tuning loop. In each iteration, it:
    a. Applies the current filter set.
    b. Re-measures the room's response.
    c. Calculates the remaining RMS error.
    d. Designs a small correction and merges it with the existing filter set.
    e. Tracks the best-performing filter set discovered so far.
4.  **Final Application:** After the loop concludes, the best-performing filter set is saved to Equalizer APO, and the UI is updated to display the final, verified results.

---

## Technology Stack

*   **Core Language:** Python (3.11+)
*   **Scientific Computing:** NumPy, SciPy (for signal processing and optimization algorithms)
*   **GUI & Visualization:** PyQt5 (QThreads for multithreading), pyqtgraph
*   **Audio I/O:** PyAudio
*   **System Integration:** Direct file I/O for Equalizer APO configuration.
*   **Packaging:** Setuptools

---

## Installation & Usage

### Prerequisites
*   Python 3.11 or newer.
*   A microphone configured as an input device.
    > **Note:** While a calibrated measurement microphone is ideal for accuracy, most standard microphones can still be used effectively to correct significant imbalances, especially in the bass and lower-midrange frequencies.
*   Equalizer APO installed (the application writes to its configuration file).

### Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/thim0o/automatic_eq_finder.git
    cd automatic_eq_finder
    ```

2.  **Create and activate a Python virtual environment:**
    ```bash
    python -m venv .venv
    # On Windows
    .venv\Scripts\activate
    ```

3.  **Install the project in editable mode:**
    This command installs the project and its dependencies, making the command-line tool available.
    ```bash
    pip install -e .
    ```

### Running the Application

With the virtual environment activated, run the application from the terminal:
```bash
automatic-eq-finder
```

---
## Future Enhancements
*   **Profile Management:** Implement functionality to save, load, and switch between different EQ profiles (e.g., for different listening positions).
*   **Expanded Compatibility:** Add support for generating EQ settings for other platforms and software, such as REW (Room EQ Wizard) or CamillaDSP.
*   **Advanced Target Curves:** Allow users to design and import their own custom target curves.

## License
This project is licensed under the MIT License.
