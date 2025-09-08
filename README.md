# Automatic EQ Equalizer

This project aims to develop an automatic equalizer that adjusts audio frequencies based on various inputs.

## Project Structure

```
automatic_eq_equalizer/
├── .gitignore
├── LICENSE
├── pyproject.toml
├── README.md
├── requirements.txt
├── setup.py
├── src/
│   ├── automatic_eq_equalizer/
│   │   ├── __init__.py
│   │   ├── audio_measurement/
│   │   │   ├── __init__.py
│   │   │   └── measurement.py
│   │   ├── cli/
│   │   │   ├── __init__.py
│   │   │   └── __main__.py
│   │   ├── eq_control/
│   │   │   ├── __init__.py
│   │   │   └── equalizer_apo.py
│   │   └── optimization/
│   │       ├── __init__.py
│   │       └── optimizer.py
│   └── main.py
└── tests/
    └── test_main.py
```

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/automatic_eq_EQUAlizer/automatic-eq-equalizer.git
    cd automatic-eq-equalizer
    ```
2.  Create a virtual environment and activate it:
    ```bash
    python -m venv .venv
    # On Windows
    .venv\Scripts\activate
    # On macOS/Linux
    source .venv/bin/activate
    ```
3.  Install the dependencies:
    ```bash
    pip install -e .
    ```

## Usage

To run the automatic EQ equalization process:
```bash
automatic-eq-equalizer
```

This will initiate the measurement, optimization, and application of EQ settings, along with real-time plotting.

## Running Tests

To run the tests, make sure you have `pytest` installed (it's included in `requirements.txt` and `setup.py`):
```bash
pytest
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.. 

TODO's:

- Use Threading / QThreads to keep the GUI responsive
- Add more tests
