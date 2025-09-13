# Automatic Room EQ Corrector

<div align="center">

![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-brightgreen)
![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)

</div>

A closed-loop acoustic control system engineered to automatically measure a room's frequency response and generate a precise parametric EQ correction. This project applies signal processing and numerical optimization to correct for acoustic distortions, producing a balanced and accurate sound signature at the listening position.

The application is a complete, end-to-end solution, from real-time audio measurement to the direct application of filter settings to system-wide equalizers like Equalizer APO.

---

## Project Overview

In any room, sound quality is directly impacted by acoustics. Reflections from walls, ceilings, and furniture create unwanted peaks and dips in the frequency response, which distorts the audio from its original source. This tool replaces the complex manual calibration process with a fully automated, data-driven system to achieve high-fidelity sound.

![Application Screenshot](Screenshot%202025-09-13%20211513.png)
*(Pictured: The application's real-time visualization dashboard after a successful optimization.)*

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
*   A measurement microphone configured as an input device.
*   Equalizer APO installed (the application writes to the default Windows config path).

### Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/automatic-eq-corrector.git
    cd automatic-eq-corrector
    ```

2.  **Create and activate a Python virtual environment:**
    ```bash
    python -m venv .venv
    # On Windows
    .venv\Scripts\activate
    ```

3.  **Install the project in editable mode:**
    This command uses the `pyproject.toml` file to install all required dependencies and makes the command-line tool available.
    ```bash
    pip install -e .
    ```

### Running the Application

With the virtual environment activated, run the application from the terminal:
```bash
automatic-eq-equalizer