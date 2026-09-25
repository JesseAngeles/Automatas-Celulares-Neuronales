# ui.TrainingPane

import os
import sys

from PyQt5.QtWidgets import (
    QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QSpinBox, QComboBox, QTextEdit, QLineEdit, QFileDialog, QCheckBox,
)
from PyQt5.QtCore import Qt, QProcess
from PyQt5.QtGui import QFont, QTextCursor

from ui.NeighbourWidget import NeighbourWidget


ACTIVATIONS = ["relu", "sigmoid", "tanh", "linear"]
ACT_LABELS  = ["ReLU", "Sigmoid", "Tanh", "Linear"]

DEFAULT_NB_3x3 = [
    [-1, -1], [0, -1], [1, -1],
    [-1,  0], [0,  0], [1,  0],
    [-1,  1], [0,  1], [1,  1],
]

TRAINING_SCRIPT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'training', 'main.py')
)


def _sep() -> QFrame:
    line = QFrame()
    line.setObjectName("sep")
    line.setFrameShape(QFrame.NoFrame)
    line.setFixedHeight(1)
    return line


def _section_label(text: str) -> QLabel:
    lbl = QLabel(text.upper())
    lbl.setObjectName("sectionTitle")
    return lbl


class TrainingPane(QWidget):

    def __init__(self):
        super().__init__()
        self._process: QProcess | None = None

        self._init_terminal()
        self._init_config_widgets()
        self._assemble_layout()

    # ── Terminal ──────────────────────────────────────────────────────────────

    def _init_terminal(self):
        self._terminal = QTextEdit()
        self._terminal.setReadOnly(True)
        f = QFont()
        f.setFamily("Monospace")
        f.setStyleHint(QFont.Monospace)
        f.setPointSize(10)
        self._terminal.setFont(f)

        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setObjectName("ghost")
        self._clear_btn.clicked.connect(self._terminal.clear)

    # ── Config widgets ────────────────────────────────────────────────────────

    def _init_config_widgets(self):
        # Data
        self._data_edit = QLineEdit()
        self._data_edit.setReadOnly(True)
        self._data_edit.setPlaceholderText("No file selected...")

        self._output_edit = QLineEdit()
        self._output_edit.setReadOnly(True)
        self._output_edit.setPlaceholderText("No file selected...")

        # Neighbourhood
        self._nb_widget = NeighbourWidget(grid_size=11)
        self._nb_widget.load_neighbourhood([DEFAULT_NB_3x3])

        # Architecture
        self._n_hidden_spin = QSpinBox()
        self._n_hidden_spin.setRange(1, 64)
        self._n_hidden_spin.setValue(1)

        self._neurons_spin = QSpinBox()
        self._neurons_spin.setRange(1, 4096)
        self._neurons_spin.setValue(64)

        # One QLineEdit per activation — user types layer numbers: "1,3,5"
        self._act_edits: dict[str, QLineEdit] = {}
        defaults = {"relu": "1", "sigmoid": "", "tanh": "", "linear": ""}
        for act in ACTIVATIONS:
            edit = QLineEdit(defaults[act])
            edit.setPlaceholderText("e.g. 1,3,5")
            self._act_edits[act] = edit

        # Output layer
        self._out_neurons_spin = QSpinBox()
        self._out_neurons_spin.setRange(1, 4096)
        self._out_neurons_spin.setValue(1)

        self._out_act_combo = QComboBox()
        for lbl in ACT_LABELS:
            self._out_act_combo.addItem(lbl)
        self._out_act_combo.setCurrentIndex(1)  # Sigmoid

        # Hyperparameters
        self._epochs_spin = QSpinBox()
        self._epochs_spin.setRange(1, 1_000_000)
        self._epochs_spin.setValue(20_000)

        self._batch_spin = QSpinBox()
        self._batch_spin.setRange(1, 1_000_000)
        self._batch_spin.setValue(8_192)

        self._lr_edit          = QLineEdit("0.001")
        self._loss_target_edit = QLineEdit("0.01")

        self._print_spin = QSpinBox()
        self._print_spin.setRange(1, 100_000)
        self._print_spin.setValue(50)

        self._seed_edit = QLineEdit()
        self._seed_edit.setPlaceholderText("Random")

        self._copy_paste_check = QCheckBox("Copy and Paste")
        self._copy_paste_check.setToolTip(
            "Exporta un CSV (epoch, loss) con el mismo nombre que el output.npz"
        )

        self._copy_paste_suffix_edit = QLineEdit()
        self._copy_paste_suffix_edit.setPlaceholderText("e.g. _2_3")
        self._copy_paste_suffix_edit.setToolTip(
            "Se añade al final del nombre del CSV: nombre + terminación + .csv"
        )

        # Buttons
        self._train_btn = QPushButton("Train")
        self._train_btn.setObjectName("primary")
        self._train_btn.setMinimumHeight(34)
        self._train_btn.clicked.connect(self._start_training)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setEnabled(False)
        self._stop_btn.setMinimumHeight(34)
        self._stop_btn.clicked.connect(self._stop_training)

    # ── Layout ────────────────────────────────────────────────────────────────

    def _assemble_layout(self):
        config_panel = QWidget()
        config_panel.setObjectName("panel")
        config_panel.setFixedWidth(760)

        config_layout = QVBoxLayout(config_panel)
        config_layout.setContentsMargins(16, 16, 16, 16)
        config_layout.setSpacing(8)

        title = QLabel("Training")
        title.setObjectName("paneTitle")
        config_layout.addWidget(title)
        config_layout.addWidget(_sep())

        # Two-column body
        body = QHBoxLayout()
        body.setSpacing(16)

        left_col  = QVBoxLayout()
        left_col.setSpacing(8)
        right_col = QVBoxLayout()
        right_col.setSpacing(8)

        # ── LEFT: DATA ────────────────────────────────────────────────
        left_col.addWidget(_section_label("Data"))

        data_browse = QPushButton("Browse...")
        data_browse.setFixedWidth(72)
        data_browse.clicked.connect(self._browse_data)
        data_row = QHBoxLayout()
        data_row.setSpacing(6)
        data_lbl = QLabel("Data (.npy/.mat)")
        data_lbl.setFixedWidth(96)
        data_row.addWidget(data_lbl)
        data_row.addWidget(self._data_edit, stretch=1)
        data_row.addWidget(data_browse)
        left_col.addLayout(data_row)

        out_browse = QPushButton("Save...")
        out_browse.setFixedWidth(72)
        out_browse.clicked.connect(self._browse_output)
        out_row = QHBoxLayout()
        out_row.setSpacing(6)
        out_lbl = QLabel("Output (.npz)")
        out_lbl.setFixedWidth(96)
        out_row.addWidget(out_lbl)
        out_row.addWidget(self._output_edit, stretch=1)
        out_row.addWidget(out_browse)
        left_col.addLayout(out_row)

        left_col.addWidget(_sep())

        # ── LEFT: NEIGHBOURHOOD ───────────────────────────────────────
        nb_hdr = QHBoxLayout()
        nb_hdr.addWidget(_section_label("Neighbourhood"))
        nb_hdr.addStretch()
        reset_nb = QPushButton("Reset")
        reset_nb.setObjectName("ghost")
        reset_nb.setFixedWidth(52)
        reset_nb.clicked.connect(self._nb_widget.reset)
        nb_hdr.addWidget(reset_nb)
        left_col.addLayout(nb_hdr)
        left_col.addWidget(self._nb_widget, alignment=Qt.AlignHCenter)
        left_col.addStretch()

        # ── RIGHT: ARCHITECTURE ───────────────────────────────────────
        right_col.addWidget(_section_label("Architecture"))

        arch_row = QHBoxLayout()
        arch_row.setSpacing(12)
        hl_lbl = QLabel("Hidden layers")
        arch_row.addWidget(hl_lbl)
        arch_row.addStretch()
        arch_row.addWidget(self._n_hidden_spin)
        neu_lbl = QLabel("Neurons")
        arch_row.addWidget(neu_lbl)
        arch_row.addWidget(self._neurons_spin)
        right_col.addLayout(arch_row)

        hint = QLabel("Enter layer numbers for each activation (e.g. 1,3,5)")
        hint.setObjectName("mutedValue")
        hint.setWordWrap(True)
        right_col.addWidget(hint)

        LABEL_W = 62
        for act, lbl_text in zip(ACTIVATIONS, ACT_LABELS):
            row = QHBoxLayout()
            row.setSpacing(8)
            lbl = QLabel(lbl_text)
            lbl.setFixedWidth(LABEL_W)
            row.addWidget(lbl)
            row.addWidget(self._act_edits[act])
            right_col.addLayout(row)

        out_row2 = QHBoxLayout()
        out_row2.setSpacing(6)
        out_lbl2 = QLabel("Output")
        out_lbl2.setFixedWidth(LABEL_W)
        out_row2.addWidget(out_lbl2)
        out_row2.addStretch()
        out_row2.addWidget(self._out_neurons_spin)
        out_row2.addWidget(self._out_act_combo)
        right_col.addLayout(out_row2)

        right_col.addWidget(_sep())

        # ── RIGHT: HYPERPARAMETERS ────────────────────────────────────
        right_col.addWidget(_section_label("Hyperparameters"))

        for label_text, widget in [
            ("Epochs",        self._epochs_spin),
            ("Batch size",    self._batch_spin),
            ("Learning rate", self._lr_edit),
            ("Loss target",   self._loss_target_edit),
            ("Print every",   self._print_spin),
            ("Seed",          self._seed_edit),
        ]:
            row = QHBoxLayout()
            row.setSpacing(6)
            row.addWidget(QLabel(label_text))
            row.addStretch()
            row.addWidget(widget)
            right_col.addLayout(row)

        cp_row = QHBoxLayout()
        cp_row.setSpacing(6)
        cp_row.addWidget(self._copy_paste_check)
        cp_row.addWidget(self._copy_paste_suffix_edit, stretch=1)
        right_col.addLayout(cp_row)

        right_col.addStretch()

        body.addLayout(left_col,  stretch=1)
        body.addLayout(right_col, stretch=1)
        config_layout.addLayout(body, stretch=1)

        # ── BUTTONS ───────────────────────────────────────────────────
        config_layout.addWidget(_sep())
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        btn_row.addWidget(self._train_btn, stretch=1)
        btn_row.addWidget(self._stop_btn)
        config_layout.addLayout(btn_row)

        # ── LOG ───────────────────────────────────────────────────────
        log_label = QLabel("Log")
        log_label.setObjectName("paneTitle")

        log_hdr = QHBoxLayout()
        log_hdr.addWidget(log_label)
        log_hdr.addStretch()
        log_hdr.addWidget(self._clear_btn)

        terminal_panel = QWidget()
        terminal_panel.setObjectName("panel")
        term_layout = QVBoxLayout(terminal_panel)
        term_layout.setContentsMargins(18, 18, 18, 18)
        term_layout.setSpacing(8)
        term_layout.addLayout(log_hdr)
        term_layout.addWidget(self._terminal)

        # ── ROOT ──────────────────────────────────────────────────────
        root = QHBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(12)
        root.addWidget(config_panel)
        root.addWidget(terminal_panel, stretch=1)

    # ── Files ─────────────────────────────────────────────────────────────────

    def _browse_data(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select data", "", "Data files (*.npy *.mat)"
        )
        if path:
            self._data_edit.setText(path)

    def _browse_output(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save weights", "", "NumPy (*.npz)")
        if path:
            if not path.endswith('.npz'):
                path += '.npz'
            self._output_edit.setText(path)

    # ── Training process ──────────────────────────────────────────────────────

    def _parse_layer_assignments(self) -> list[str] | None:
        n = self._n_hidden_spin.value()
        neurons = self._neurons_spin.value()

        layer_act: dict[int, str] = {}
        for act in ACTIVATIONS:
            text = self._act_edits[act].text().strip()
            if not text:
                continue
            for part in text.split(','):
                part = part.strip()
                if not part:
                    continue
                try:
                    layer_num = int(part)
                except ValueError:
                    self._log(f"Error: '{part}' is not a valid layer number\n")
                    return None
                if not (1 <= layer_num <= n):
                    self._log(f"Error: layer {layer_num} out of range (1–{n})\n")
                    return None
                if layer_num in layer_act:
                    self._log(f"Error: layer {layer_num} assigned to multiple activations\n")
                    return None
                layer_act[layer_num] = act

        specs = []
        for i in range(1, n + 1):
            act = layer_act.get(i, 'relu')
            specs.append(f"{neurons}:{act}")
        return specs

    def _build_command(self) -> list | None:
        data_path = self._data_edit.text().strip()
        if not data_path:
            self._log("Error: select a data file (.npy/.mat)\n")
            return None

        output_path = self._output_edit.text().strip() or "weights.npz"

        groups = self._nb_widget.get_neighbourhood()
        if not groups or not groups[0]:
            self._log("Error: neighbourhood is empty — select at least one cell\n")
            return None
        nb_str = " ".join(f"{dx},{dy}" for dx, dy in groups[0])

        specs = self._parse_layer_assignments()
        if specs is None:
            return None
        layers_str = " ".join(specs)

        out_size = self._out_neurons_spin.value()
        out_act  = ACTIVATIONS[self._out_act_combo.currentIndex()]

        try:
            lr          = float(self._lr_edit.text())
            loss_target = float(self._loss_target_edit.text())
        except ValueError:
            self._log("Error: invalid learning rate or loss target\n")
            return None

        cmd = [
            "-u", TRAINING_SCRIPT,
            "--data",          data_path,
            "--output",        output_path,
            "--neighbourhood", nb_str,
            "--layers",        layers_str,
            "--out-size",      str(out_size),
            "--out-act",       out_act,
            "--epochs",        str(self._epochs_spin.value()),
            "--batch",         str(self._batch_spin.value()),
            "--lr",            str(lr),
            "--loss-target",   str(loss_target),
            "--print-every",   str(self._print_spin.value()),
        ]

        seed_text = self._seed_edit.text().strip()
        if seed_text:
            try:
                seed_val = int(seed_text)
            except ValueError:
                self._log(f"Error: '{seed_text}' is not a valid seed (must be an integer)\n")
                return None
            cmd += ["--seed", str(seed_val)]

        if self._copy_paste_check.isChecked():
            suffix   = self._copy_paste_suffix_edit.text().strip()
            csv_path = os.path.splitext(output_path)[0] + suffix + ".csv"
            cmd += ["--log-csv", csv_path]

        return cmd

    def _start_training(self):
        cmd = self._build_command()
        if cmd is None:
            return

        self._terminal.clear()
        self._log(f"[{sys.executable} {' '.join(cmd)}]\n\n")

        self._process = QProcess(self)
        self._process.readyReadStandardOutput.connect(self._on_stdout)
        self._process.readyReadStandardError.connect(self._on_stderr)
        self._process.finished.connect(self._on_finished)
        self._process.start(sys.executable, cmd)

        self._train_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)

    def _stop_training(self):
        if self._process and self._process.state() != QProcess.NotRunning:
            self._process.terminate()

    def _on_stdout(self):
        data = self._process.readAllStandardOutput().data().decode('utf-8', errors='replace')
        self._log(data)

    def _on_stderr(self):
        data = self._process.readAllStandardError().data().decode('utf-8', errors='replace')
        self._log(data)

    def _on_finished(self, exit_code, _):
        self._train_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._log(f"\n[Process finished — code {exit_code}]\n")
        self._process = None

    def _log(self, text: str):
        self._terminal.moveCursor(QTextCursor.End)
        self._terminal.insertPlainText(text)
        self._terminal.ensureCursorVisible()
