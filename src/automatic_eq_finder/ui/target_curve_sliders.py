from PyQt5 import QtWidgets, QtCore


class TargetCurveSliders(QtWidgets.QGroupBox):
    """
    A reusable widget for adjusting target EQ curve parameters via sliders.
    Emits a signal when any parameter changes.
    """

    parametersChanged = QtCore.pyqtSignal(
        dict
    )  # emits {'bass_gain_db': ..., 'corner_freq': ..., 'tilt_db_per_decade': ...}

    def __init__(self, bass_gain_db, corner_freq, tilt_db_per_decade, parent=None):
        super().__init__("Target Curve Adjustment", parent)
        self.setMaximumHeight(80)

        self._bass_gain_db = bass_gain_db
        self._corner_freq = corner_freq
        self._tilt_db_per_decade = tilt_db_per_decade

        layout = QtWidgets.QHBoxLayout()
        self.setLayout(layout)

        # Add sliders
        self._add_slider(
            layout,
            "Bass Gain (dB)",
            0.0,
            18.0,
            bass_gain_db,
            10,
            "bass_gain_db",
            "{:.1f} dB",
        )
        self._add_slider(
            layout,
            "Tilt (dB/dec)",
            -15.0,
            3.0,
            tilt_db_per_decade,
            100,
            "tilt_db_per_decade",
            "{:.2f}",
        )
        self._add_slider(
            layout,
            "Corner Freq (Hz)",
            30.0,
            230.0,
            corner_freq,
            1,
            "corner_freq",
            "{:.0f} Hz",
        )

    def _add_slider(
        self,
        parent_layout,
        label_text,
        min_val,
        max_val,
        default_val,
        scale,
        param_name,
        format_str,
    ):
        container = QtWidgets.QVBoxLayout()

        label_row = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel(label_text)
        value_label = QtWidgets.QLabel(format_str.format(default_val))
        value_label.setFixedWidth(60)
        label_row.addWidget(label)
        label_row.addStretch(1)
        label_row.addWidget(value_label)

        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setRange(int(min_val * scale), int(max_val * scale))
        slider.setValue(int(default_val * scale))
        slider.setSingleStep(1)

        def on_change(value):
            real_value = value / scale
            value_label.setText(format_str.format(real_value))
            setattr(self, f"_{param_name}", real_value)
            self.parametersChanged.emit(
                {
                    "bass_gain_db": self._bass_gain_db,
                    "corner_freq": self._corner_freq,
                    "tilt_db_per_decade": self._tilt_db_per_decade,
                }
            )

        slider.valueChanged.connect(on_change)

        container.addLayout(label_row)
        container.addWidget(slider)
        parent_layout.addLayout(container)
