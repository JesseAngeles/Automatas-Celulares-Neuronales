# ui.ControlPane

from PyQt5.QtWidgets import (
    QWidget, QPushButton, QSlider, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QDoubleSpinBox, QSpinBox
)
from PyQt5.QtCore import pyqtSignal, Qt


def _separator() -> QFrame:
    line = QFrame()
    line.setObjectName("sep")
    line.setFrameShape(QFrame.NoFrame)
    line.setFixedHeight(1)
    return line


def _section_label(text: str) -> QLabel:
    lbl = QLabel(text.upper())
    lbl.setObjectName("sectionTitle")
    return lbl


class ControlPane(QWidget):
    play_clicked    = pyqtSignal()
    step_clicked    = pyqtSignal()
    reset_clicked   = pyqtSignal()
    speed_moved     = pyqtSignal(int)
    timeline_jumped = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.setObjectName("panel")

        # ── Header: título + contador de pasos ───────────────────────
        title = QLabel("Simulation")
        title.setObjectName("paneTitle")

        self.step_label = QLabel("Step 0")
        self.step_label.setObjectName("stepCounter")
        self.step_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        header = QHBoxLayout()
        header.addWidget(title)
        header.addStretch()
        header.addWidget(self.step_label)

        # ── Timeline ──────────────────────────────────────────────────
        self.timeline_slider = QSlider(Qt.Horizontal)
        self.timeline_slider.setMinimum(0)
        self.timeline_slider.setMaximum(0)
        self.timeline_slider.setValue(0)
        self.timeline_slider.setEnabled(False)
        self.timeline_slider.setObjectName("timelineSlider")

        # ── Playback ──────────────────────────────────────────────────
        self.backstep_button = QPushButton("◀")
        self.play_button     = QPushButton("▶  Play")
        self.step_button     = QPushButton("▶|")

        self.play_button.setObjectName("primary")
        self.play_button.setMinimumHeight(34)
        self.backstep_button.setFixedWidth(46)
        self.step_button.setFixedWidth(46)
        self.backstep_button.setMinimumHeight(34)
        self.step_button.setMinimumHeight(34)

        playback_row = QHBoxLayout()
        playback_row.setSpacing(6)
        playback_row.addWidget(self.backstep_button)
        playback_row.addWidget(self.play_button, stretch=1)
        playback_row.addWidget(self.step_button)

        # ── Speed ─────────────────────────────────────────────────────
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(1000)
        self.speed_slider.setValue(750)

        self._speed_value = QLabel("750")
        self._speed_value.setObjectName("mutedValue")

        speed_head = QHBoxLayout()
        speed_head.addWidget(_section_label("Speed"))
        speed_head.addStretch()
        speed_head.addWidget(self._speed_value)

        self.speed_slider.valueChanged.connect(
            lambda v: self._speed_value.setText(str(v))
        )

        # ── Stochastic ────────────────────────────────────────────────
        self.stochastic_spin = QDoubleSpinBox()
        self.stochastic_spin.setRange(0.01, 1.0)
        self.stochastic_spin.setSingleStep(0.01)
        self.stochastic_spin.setDecimals(2)
        self.stochastic_spin.setValue(1.0)

        # p = 1.0 → actualización síncrona clásica (sin efecto estocástico)
        self._stochastic_state = QLabel("deterministic")
        self._stochastic_state.setObjectName("mutedValue")

        stochastic_head = QHBoxLayout()
        stochastic_head.addWidget(_section_label("Stochastic p"))
        stochastic_head.addStretch()
        stochastic_head.addWidget(self._stochastic_state)

        self.stochastic_spin.valueChanged.connect(
            lambda v: self._stochastic_state.setText(
                "deterministic" if v >= 1.0 else "stochastic"
            )
        )

        # ── Grid ──────────────────────────────────────────────────────
        self.erase_button  = QPushButton("Erase")
        self.random_button = QPushButton("Random")

        grid_row = QHBoxLayout()
        grid_row.setSpacing(6)
        grid_row.addWidget(self.erase_button)
        grid_row.addWidget(self.random_button)

        # ── Record ────────────────────────────────────────────────────
        self.record_button = QPushButton("Record N steps")

        # ── Files ─────────────────────────────────────────────────────
        self.import_button = QPushButton("Import")
        self.export_button = QPushButton("Export")
        self.import_button.setObjectName("ghost")
        self.export_button.setObjectName("ghost")

        file_row = QHBoxLayout()
        file_row.setSpacing(6)
        file_row.addWidget(self.import_button)
        file_row.addWidget(self.export_button)

        # ── NCA ───────────────────────────────────────────────────────
        self.load_nca_button = QPushButton("Load NCA weights")
        self.mode_button = QPushButton("Mode: CA")
        self.mode_button.setObjectName("ghost")
        self.mode_button.setEnabled(False)
        self.continuous_button = QPushButton("Output: Discrete")
        self.continuous_button.setObjectName("ghost")
        self.continuous_button.setEnabled(False)

        # Selector de canal: solo habilitado con un NCA multi-canal cargado
        self.channel_spin = QSpinBox()
        self.channel_spin.setRange(0, 0)
        self.channel_spin.setEnabled(False)

        channel_row = QHBoxLayout()
        channel_row.setSpacing(6)
        channel_row.addWidget(QLabel("Channel"))
        channel_row.addStretch()
        channel_row.addWidget(self.channel_spin)
        self._channel_row = channel_row

        # ── Layout principal ──────────────────────────────────────────
        layout = QVBoxLayout()
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(10)

        layout.addLayout(header)
        layout.addWidget(self.timeline_slider)
        layout.addWidget(_separator())

        layout.addLayout(playback_row)
        layout.addSpacing(4)
        layout.addLayout(speed_head)
        layout.addWidget(self.speed_slider)
        layout.addWidget(_separator())

        layout.addLayout(stochastic_head)
        layout.addWidget(self.stochastic_spin)
        layout.addWidget(_separator())

        layout.addWidget(_section_label("Grid"))
        layout.addLayout(grid_row)
        layout.addWidget(_separator())

        layout.addWidget(_section_label("Record"))
        layout.addWidget(self.record_button)
        layout.addWidget(_separator())

        layout.addWidget(_section_label("Files"))
        layout.addLayout(file_row)
        layout.addWidget(_separator())

        layout.addWidget(_section_label("NCA"))
        layout.addWidget(self.load_nca_button)
        layout.addWidget(self.mode_button)
        layout.addWidget(self.continuous_button)
        layout.addLayout(self._channel_row)

        layout.addStretch()
        self.setLayout(layout)

        # ── Signals ───────────────────────────────────────────────────
        self.play_button.clicked.connect(self.play_clicked.emit)
        self.step_button.clicked.connect(self.step_clicked.emit)
        self.erase_button.clicked.connect(self.reset_clicked.emit)
        self.speed_slider.valueChanged.connect(self.speed_moved.emit)
        self.timeline_slider.valueChanged.connect(self.timeline_jumped.emit)

    # ------------------------------------------------------------------
    def update_play_button(self, is_running: bool):
        self.play_button.setText("⏸  Pause" if is_running else "▶  Play")

    def update_step_label(self, n: int):
        self.step_label.setText(f"Step {n}")

    def update_continuous_button(self, continuous: bool, available: bool):
        self.continuous_button.setText(
            "Output: Continuous" if continuous else "Output: Discrete"
        )
        self.continuous_button.setEnabled(available)

    def update_channel_selector(self, channels: int):
        self.channel_spin.blockSignals(True)
        self.channel_spin.setRange(0, max(0, channels - 1))
        self.channel_spin.setValue(0)
        self.channel_spin.blockSignals(False)
        self.channel_spin.setEnabled(channels > 1)

    def update_timeline(self, current: int, total: int):
        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setMaximum(total)
        self.timeline_slider.setValue(current)
        self.timeline_slider.setEnabled(total > 0)
        self.timeline_slider.blockSignals(False)

    def connect_controller(self, controller):
        self.step_button.clicked.connect(controller.on_step_button)
        self.play_button.clicked.connect(controller.on_play_button)
        self.backstep_button.clicked.connect(controller.on_backstep_button)
        self.erase_button.clicked.connect(controller.on_erase_button)
        self.random_button.clicked.connect(controller.on_random_button)
        self.record_button.clicked.connect(controller.on_record_button)
        self.import_button.clicked.connect(controller.on_import_button)
        self.export_button.clicked.connect(controller.on_export_button)
        self.load_nca_button.clicked.connect(controller.on_load_nca_button)
        self.mode_button.clicked.connect(controller.on_mode_toggle)
        self.continuous_button.clicked.connect(controller.on_continuous_toggle)
        self.stochastic_spin.valueChanged.connect(controller.on_stochastic_p_change)
        self.channel_spin.valueChanged.connect(controller.on_display_channel_change)
        self.timeline_jumped.connect(controller.on_timeline_jump)
        controller._on_timeline_changed = self.update_timeline
        controller._on_continuous_changed = self.update_continuous_button
        controller._on_channels_changed = self.update_channel_selector
