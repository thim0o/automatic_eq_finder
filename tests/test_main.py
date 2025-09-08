import pytest
from automatic_eq_equalizer.cli.__main__ import main

def test_main_function_prints_hello(capsys):
    # This test assumes that the main function in cli/__main__.py
    # still prints "Hello, Automatic EQ Equalizer!" as its initial action.
    main()
    captured = capsys.readouterr()
    assert "Hello, Automatic EQ Equalizer!" in captured.out