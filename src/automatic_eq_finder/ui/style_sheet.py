# This makes the entire UI, including buttons and backgrounds, dark.

DARK_STYLESHEET = """
    QWidget {
        background-color: #1e1e1e;
        color: #dcdcdc;
        font-family: Segoe UI, sans-serif;
        font-size: 14pt;
    }
    QMainWindow {
        background-color: #1e1e1e;
    }
    QGroupBox {
        background-color: #2d2d2d;
        border: 1px solid #444444;
        border-radius: 5px;
        margin-top: 1ex; /* leave space at the top for the title */
        font-weight: bold;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top center; /* position at the top center */
        padding: 0 3px;
        background-color: #2d2d2d;
        color: #dcdcdc;
    }
    QCheckBox {
        color: #dcdcdc;
    }
    QCheckBox::indicator {
        width: 13px;
        height: 13px;
        border: 1px solid #555555;
        border-radius: 3px;
    }
    QCheckBox::indicator:checked {
        background-color: #55aaff; /* A pleasant blue for checked state */
        border: 1px solid #55aaff;
    }
"""
