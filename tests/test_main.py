import pytest
from automatic_eq_finder.cli.__main__ import main

def test_main_function_prints_hello(capsys):
    # This test assumes that the main function in cli/__main__.py
    main()
    captured = capsys.readouterr()
    assert "Launching Automatic EQ Equalizer UI..." in captured.out