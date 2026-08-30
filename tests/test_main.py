import pytest
from unittest.mock import MagicMock, patch
from automatic_eq_finder.cli.__main__ import main

def test_main_function_prints_hello(capsys):
    app = MagicMock()
    plotter = MagicMock()
    worker = MagicMock()

    with patch("automatic_eq_finder.cli.__main__.QtWidgets.QApplication", return_value=app), \
            patch("automatic_eq_finder.cli.__main__.RealTimePlotter", return_value=plotter), \
            patch("automatic_eq_finder.cli.__main__.AutoEQIterative", return_value=worker):
        main()

    captured = capsys.readouterr()
    assert "Launching Automatic EQ Equalizer UI..." in captured.out
    app.exec_.assert_called_once()
