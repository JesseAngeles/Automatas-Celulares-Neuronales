# ui.TrainingPane

import math
import os
import sys

from PyQt5.QtWidgets import (
    QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QFrame,
    QSpinBox, QComboBox, QTextEdit, QLineEdit, QFileDialog, QCheckBox, QListWidget,
    QAbstractItemView, QScrollArea, QButtonGroup, QRadioButton,
)
from PyQt5.QtCore import Qt, QProcess, QPointF
from PyQt5.QtGui import QFont, QTextCursor, QPainter, QPen, QColor, QPainterPath

from ui.NeighbourWidget import NeighbourWidget


ACTIVATIONS = ["relu", "sigmoid", "tanh", "linear"]
ACT_LABELS  = ["ReLU", "Sigmoid", "Tanh", "Linear"]

# (valor de --lr-schedule, texto en el selector)
SCHEDULES = [
    ("exp",     "Exponential"),
    ("cosine",  "Cosine"),
    ("plateau", "On plateau"),
    ("none",    "Constant"),
]

DEFAULT_NB_3x3 = [
    [-1, -1], [0, -1], [1, -1],
    [-1,  0], [0,  0], [1,  0],
    [-1,  1], [0,  1], [1,  1],
]

TRAINING_SCRIPT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'training', 'scripts', 'main.py')
)

ACCENT = QColor("#3478f6")
MUTED  = QColor("#8a8a92")
GRID   = QColor("#ececf0")


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


def _muted(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("mutedValue")
    lbl.setWordWrap(True)
    return lbl


def _square(radius: int) -> list:
    return [[dx, dy] for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)]


class _Form(QGridLayout):
    """Etiquetas a la izquierda y campos alineados en una columna."""

    def __init__(self):
        super().__init__()
        self.setHorizontalSpacing(10)
        self.setVerticalSpacing(6)
        self.setColumnMinimumWidth(0, 92)
        self.setColumnStretch(1, 1)
        self._row = 0

    def add(self, label: str, *widgets, stretch_last: bool = True):
        lbl = QLabel(label)
        self.addWidget(lbl, self._row, 0)
        if len(widgets) == 1:
            self.addWidget(widgets[0], self._row, 1)
        else:
            box = QHBoxLayout()
            box.setSpacing(6)
            for w in widgets:
                box.addWidget(w)
            if not stretch_last:
                box.addStretch()
            self.addLayout(box, self._row, 1)
        self._row += 1
        return lbl


class LRPreview(QWidget):
    """Curva del learning rate por época con el decaimiento elegido (eje vertical logarítmico)."""

    def __init__(self):
        super().__init__()
        self.setFixedHeight(74)
        self._schedule, self._lr, self._lr_min, self._epochs = "exp", 1e-3, 1e-6, 20_000
        self._factor = 0.5

    def update_curve(self, schedule: str, lr: float, lr_min: float, epochs: int, factor: float):
        self._schedule, self._lr, self._lr_min, self._epochs = schedule, lr, lr_min, max(epochs, 1)
        self._factor = factor
        self.update()

    def _value(self, x: float) -> float:
        """lr en la fracción x ∈ [0, 1] del entrenamiento."""
        lr, lo = self._lr, min(self._lr_min, self._lr)
        if self._schedule == "exp":
            return lr * (lo / lr) ** x
        if self._schedule == "cosine":
            return lo + (lr - lo) * 0.5 * (1 + math.cos(math.pi * x))
        if self._schedule == "plateau":
            # ilustrativo: los escalones dependen de la pérdida, aquí se reparten parejos
            steps = max(1, min(8, int(math.log(lo / lr) / math.log(self._factor)))) if lo < lr else 0
            k = min(steps, int(x * (steps + 1)))
            return max(lo, lr * self._factor ** k)
        return lr

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        left, right, top, bottom = 40, 4, 6, 16

        lr, lo = self._lr, min(self._lr_min, self._lr)
        if self._schedule == "none" or lo <= 0 or lr <= 0:
            lo = lr / 10 if lr > 0 else 1e-6
        log_hi, log_lo = math.log10(lr), math.log10(lo)
        span = log_hi - log_lo or 1

        def pt(x: float, v: float) -> QPointF:
            y = (log_hi - math.log10(max(v, 1e-300))) / span
            return QPointF(left + x * (w - left - right), top + y * (h - top - bottom))

        p.setPen(QPen(GRID, 1))
        p.drawLine(QPointF(left, h - bottom), QPointF(w - right, h - bottom))

        path = QPainterPath(pt(0, self._value(0)))
        n = 160
        for i in range(1, n + 1):
            x = i / n
            v = self._value(x)
            if self._schedule == "plateau":
                path.lineTo(pt(x, self._value(x - 1e-9)))
            path.lineTo(pt(x, v))
        pen = QPen(ACCENT, 2)
        if self._schedule == "plateau":
            pen.setStyle(Qt.DashLine)
        p.setPen(pen)
        p.drawPath(path)

        p.setPen(MUTED)
        f = p.font()
        f.setPointSizeF(8.5)
        p.setFont(f)
        p.drawText(left, h - 3, "0")
        p.drawText(0, top - 4, left - 6, 14, Qt.AlignRight | Qt.AlignTop, f"{lr:g}")
        if self._schedule != "none":
            p.drawText(0, h - bottom - 10, left - 6, 14, Qt.AlignRight | Qt.AlignBottom, f"{lo:g}")
        end = f"{self._epochs:,} epochs" if self._schedule != "plateau" else "depends on the loss"
        p.drawText(0, h - 14, w - right, 14, Qt.AlignRight | Qt.AlignBottom, end)
        p.end()


class TrainingPane(QWidget):

    def __init__(self):
        super().__init__()
        self._process: QProcess | None = None

        self._init_terminal()
        self._init_config_widgets()
        self._assemble_layout()
        self._sync_schedule()
        self._sync_input()

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
        self._data_list = QListWidget()
        self._data_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._data_list.setFixedHeight(64)
        self._data_list.setTextElideMode(Qt.ElideLeft)
        self._data_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._data_list.setToolTip("Una o varias evoluciones; sus transiciones se juntan al entrenar")

        self._data_add_btn = QPushButton("Add...")
        self._data_add_btn.clicked.connect(self._browse_data)
        self._data_remove_btn = QPushButton("Remove")
        self._data_remove_btn.setObjectName("ghost")
        self._data_remove_btn.clicked.connect(self._remove_data)

        self._mat_key_edit = QLineEdit()
        self._mat_key_edit.setPlaceholderText("Automatic")
        self._mat_key_edit.setToolTip("Variable a leer si el .mat tiene varias matrices")

        self._output_edit = QLineEdit()
        self._output_edit.setReadOnly(True)
        self._output_edit.setPlaceholderText("weights.npz")

        self._loss_csv_check = QCheckBox("Save loss CSV")
        self._loss_csv_check.setToolTip(
            "Exporta un CSV (epoch, loss, lr) con el mismo nombre que el .npz de salida"
        )
        self._loss_csv_suffix_edit = QLineEdit()
        self._loss_csv_suffix_edit.setPlaceholderText("Name suffix, e.g. _2_3")
        self._loss_csv_suffix_edit.setToolTip("Se añade al nombre del CSV: nombre + terminación + .csv")
        self._loss_csv_check.toggled.connect(self._loss_csv_suffix_edit.setEnabled)
        self._loss_csv_suffix_edit.setEnabled(False)

        # Neighbourhood
        self._nb_widget = NeighbourWidget(grid_size=11)
        self._nb_widget.load_neighbourhood([DEFAULT_NB_3x3])

        self._boundary_group = QButtonGroup(self)
        self._periodic_radio = QRadioButton("Periodic")
        self._open_radio = QRadioButton("Open")
        self._open_radio.setToolTip("Frontera abierta: los vecinos fuera del espacio valen 0")
        self._periodic_radio.setChecked(True)
        self._boundary_group.addButton(self._periodic_radio)
        self._boundary_group.addButton(self._open_radio)

        # Network: input
        self._input_group = QButtonGroup(self)
        self._raw_radio = QRadioButton("State value")
        self._raw_radio.setToolTip("Cada vecino entra como su número de estado")
        self._one_hot_radio = QRadioButton("One-hot")
        self._one_hot_radio.setToolTip(
            "Codifica cada vecino en one-hot: vecinos × estados entradas en la primera capa"
        )
        self._raw_radio.setChecked(True)
        self._input_group.addButton(self._raw_radio)
        self._input_group.addButton(self._one_hot_radio)
        self._one_hot_radio.toggled.connect(self._sync_input)

        self._one_hot_states_spin = QSpinBox()
        self._one_hot_states_spin.setRange(1, 1024)
        self._one_hot_states_spin.setSpecialValueText("From data")
        self._one_hot_states_spin.setValue(1)
        self._one_hot_states_spin.setToolTip("Número de estados; 'From data' usa el máximo de la evolución + 1")

        # Network: hidden layers
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
            edit.setPlaceholderText("Layers, e.g. 1,3,5")
            self._act_edits[act] = edit

        # Network: output layer
        self._out_neurons_spin = QSpinBox()
        self._out_neurons_spin.setRange(1, 4096)
        self._out_neurons_spin.setValue(1)
        self._out_neurons_spin.setToolTip("1 neurona: 2 estados (BCE). Más de 1: una por estado (cross-entropy)")

        self._out_act_combo = QComboBox()
        for lbl in ACT_LABELS:
            self._out_act_combo.addItem(lbl)
        self._out_act_combo.setCurrentIndex(1)  # Sigmoid

        # Training
        self._epochs_spin = QSpinBox()
        self._epochs_spin.setRange(1, 1_000_000)
        self._epochs_spin.setValue(20_000)
        self._epochs_spin.setGroupSeparatorShown(True)

        self._batch_spin = QSpinBox()
        self._batch_spin.setRange(1, 1_000_000)
        self._batch_spin.setValue(8_192)
        self._batch_spin.setGroupSeparatorShown(True)

        self._loss_target_edit = QLineEdit("0.01")
        self._loss_target_edit.setToolTip("El entrenamiento se detiene cuando la pérdida baja de este valor")

        self._print_spin = QSpinBox()
        self._print_spin.setRange(1, 100_000)
        self._print_spin.setValue(50)

        self._seed_edit = QLineEdit()
        self._seed_edit.setPlaceholderText("Random")

        # Learning rate
        self._lr_edit = QLineEdit("0.001")
        self._lr_min_edit = QLineEdit("1e-6")
        self._lr_min_edit.setToolTip("Learning rate final (exp, cosine) o mínimo (plateau)")

        self._schedule_combo = QComboBox()
        for _, text in SCHEDULES:
            self._schedule_combo.addItem(text)
        self._schedule_combo.setToolTip(
            "Exponential y Cosine llegan al valor final en la última época.\n"
            "On plateau reduce el learning rate cuando la pérdida deja de mejorar.\n"
            "Constant no lo cambia."
        )

        self._patience_spin = QSpinBox()
        self._patience_spin.setRange(1, 100_000)
        self._patience_spin.setValue(20)
        self._patience_spin.setSuffix(" epochs")

        self._factor_edit = QLineEdit("0.5")
        self._factor_edit.setToolTip("Se multiplica el learning rate por este factor en cada reducción")

        self._schedule_hint = _muted("")
        self._lr_preview = LRPreview()

        for w in (self._lr_edit, self._lr_min_edit, self._factor_edit):
            w.textChanged.connect(self._sync_schedule)
        self._schedule_combo.currentIndexChanged.connect(self._sync_schedule)
        self._epochs_spin.valueChanged.connect(self._sync_schedule)

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
        config_panel.setFixedWidth(820)

        config_layout = QVBoxLayout(config_panel)
        config_layout.setContentsMargins(16, 16, 16, 16)
        config_layout.setSpacing(8)

        title = QLabel("Training")
        title.setObjectName("paneTitle")
        config_layout.addWidget(title)
        config_layout.addWidget(_sep())

        body = QHBoxLayout()
        body.setSpacing(24)
        left_col = QVBoxLayout()
        left_col.setSpacing(8)
        right_col = QVBoxLayout()
        right_col.setSpacing(8)

        # ── LEFT: DATA ────────────────────────────────────────────────
        left_col.addWidget(_section_label("Data"))
        data_form = _Form()
        data_btns = QVBoxLayout()
        data_btns.setSpacing(6)
        data_btns.addWidget(self._data_add_btn)
        data_btns.addWidget(self._data_remove_btn)
        data_btns.addStretch()
        data_box = QHBoxLayout()
        data_box.setSpacing(6)
        data_box.addWidget(self._data_list, stretch=1)
        data_box.addLayout(data_btns)
        lbl = QLabel("Evolutions")
        lbl.setContentsMargins(0, 4, 0, 0)
        data_form.addWidget(lbl, 0, 0, Qt.AlignTop)
        data_form.addLayout(data_box, 0, 1)
        data_form._row = 1
        data_form.add(".mat variable", self._mat_key_edit)

        out_browse = QPushButton("Save as...")
        out_browse.clicked.connect(self._browse_output)
        data_form.add("Weights", self._output_edit, out_browse)
        data_form.add("", self._loss_csv_check, self._loss_csv_suffix_edit)
        left_col.addLayout(data_form)

        left_col.addSpacing(6)
        left_col.addWidget(_sep())

        # ── LEFT: NEIGHBOURHOOD ───────────────────────────────────────
        nb_hdr = QHBoxLayout()
        nb_hdr.addWidget(_section_label("Neighbourhood"))
        nb_hdr.addStretch()
        for r in (1, 2, 3, 4):
            b = QPushButton(f"{2 * r + 1}×{2 * r + 1}")
            b.setObjectName("ghost")
            b.setFixedWidth(44)
            b.setToolTip(f"Vecindad cuadrada de radio {r}")
            b.clicked.connect(lambda _, r=r: self._nb_widget.load_neighbourhood([_square(r)]))
            nb_hdr.addWidget(b)
        reset_nb = QPushButton("Clear")
        reset_nb.setObjectName("ghost")
        reset_nb.setFixedWidth(52)
        reset_nb.clicked.connect(self._nb_widget.reset)
        nb_hdr.addWidget(reset_nb)
        left_col.addLayout(nb_hdr)
        left_col.addWidget(self._nb_widget, alignment=Qt.AlignHCenter)

        nb_form = _Form()
        nb_form.add("Boundary", self._periodic_radio, self._open_radio, stretch_last=False)
        left_col.addLayout(nb_form)

        left_col.addSpacing(6)
        left_col.addWidget(_sep())

        # ── LEFT: TRAINING ───────────────────────────────────────────
        left_col.addWidget(_section_label("Training"))
        train_form = _Form()
        train_form.add("Epochs", self._epochs_spin)
        train_form.add("Batch size", self._batch_spin)
        train_form.add("Loss target", self._loss_target_edit)
        train_form.add("Seed", self._seed_edit)
        train_form.add("Print every", self._print_spin)
        left_col.addLayout(train_form)

        left_col.addStretch()

        # ── RIGHT: NETWORK ────────────────────────────────────────────
        right_col.addWidget(_section_label("Network"))
        net_form = _Form()
        self._states_lbl = QLabel("States")
        net_form.add("Input", self._raw_radio, self._one_hot_radio, stretch_last=False)
        net_form.addWidget(self._states_lbl, net_form._row, 0)
        net_form.addWidget(self._one_hot_states_spin, net_form._row, 1)
        net_form._row += 1
        net_form.add("Hidden layers", self._n_hidden_spin, QLabel("of"), self._neurons_spin,
                     QLabel("neurons"), stretch_last=False)
        for act, lbl_text in zip(ACTIVATIONS, ACT_LABELS):
            net_form.add(lbl_text, self._act_edits[act])
        net_form.add("Output", self._out_neurons_spin, QLabel("neurons,"), self._out_act_combo,
                     stretch_last=False)
        right_col.addLayout(net_form)
        right_col.addWidget(_muted("Layers without an activation use ReLU."))

        right_col.addSpacing(6)
        right_col.addWidget(_sep())

        # ── RIGHT: LEARNING RATE ──────────────────────────────────────
        right_col.addWidget(_section_label("Learning rate"))
        lr_form = _Form()
        lr_form.add("Schedule", self._schedule_combo)
        lr_form.add("Initial", self._lr_edit)
        self._lr_min_lbl = lr_form.add("Final", self._lr_min_edit)
        self._patience_lbl = lr_form.add("Patience", self._patience_spin)
        self._factor_lbl = lr_form.add("Factor", self._factor_edit)
        right_col.addLayout(lr_form)
        right_col.addWidget(self._lr_preview)
        right_col.addWidget(self._schedule_hint)
        right_col.addStretch()

        body.addLayout(left_col, stretch=1)
        body.addLayout(right_col, stretch=1)
        config_layout.addLayout(body, stretch=1)

        # ── BUTTONS ───────────────────────────────────────────────────
        config_layout.addWidget(_sep())
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        btn_row.addWidget(self._train_btn, stretch=1)
        btn_row.addWidget(self._stop_btn)
        config_layout.addLayout(btn_row)

        scroll = QScrollArea()
        scroll.setWidget(config_panel)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFixedWidth(config_panel.width() + 12)

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
        root.addWidget(scroll)
        root.addWidget(terminal_panel, stretch=1)

    # ── Dependent fields ──────────────────────────────────────────────────────

    def _schedule_value(self) -> str:
        return SCHEDULES[self._schedule_combo.currentIndex()][0]

    def _sync_schedule(self):
        schedule = self._schedule_value()
        plateau = schedule == "plateau"
        for w in (self._patience_lbl, self._patience_spin, self._factor_lbl, self._factor_edit):
            w.setVisible(plateau)
        self._lr_min_lbl.setText("Minimum" if plateau else "Final")
        self._lr_min_lbl.setEnabled(schedule != "none")
        self._lr_min_edit.setEnabled(schedule != "none")

        hints = {
            "exp": "Decays exponentially and reaches the final value on the last epoch. "
                   "If the loss target is reached earlier, training stops at a higher rate.",
            "cosine": "Follows half a cosine and reaches the final value on the last epoch.",
            "plateau": "Multiplies the rate by the factor when the loss has not improved for "
                       "the patience epochs, without going below the minimum.",
            "none": "Keeps the initial rate for the whole training.",
        }
        self._schedule_hint.setText(hints[schedule])

        try:
            lr = float(self._lr_edit.text())
            lr_min = float(self._lr_min_edit.text())
            factor = float(self._factor_edit.text())
        except ValueError:
            return
        if lr > 0 and lr_min > 0 and 0 < factor < 1:
            self._lr_preview.update_curve(schedule, lr, lr_min, self._epochs_spin.value(), factor)

    def _sync_input(self):
        one_hot = self._one_hot_radio.isChecked()
        self._states_lbl.setVisible(one_hot)
        self._one_hot_states_spin.setVisible(one_hot)

    # ── Files ─────────────────────────────────────────────────────────────────

    def _browse_data(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select evolutions", "", "Data files (*.npy *.mat)"
        )
        existing = {self._data_list.item(i).text() for i in range(self._data_list.count())}
        for path in paths:
            if path not in existing:
                self._data_list.addItem(path)
                self._data_list.item(self._data_list.count() - 1).setToolTip(path)

    def _remove_data(self):
        for item in self._data_list.selectedItems():
            self._data_list.takeItem(self._data_list.row(item))

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

    def _float_field(self, edit: QLineEdit, name: str) -> float | None:
        try:
            return float(edit.text())
        except ValueError:
            self._log(f"Error: '{edit.text()}' is not a valid {name}\n")
            return None

    def _build_command(self) -> list | None:
        data_paths = [self._data_list.item(i).text() for i in range(self._data_list.count())]
        if not data_paths:
            self._log("Error: add at least one evolution (.npy/.mat)\n")
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

        lr          = self._float_field(self._lr_edit, "learning rate")
        lr_min      = self._float_field(self._lr_min_edit, "final learning rate")
        loss_target = self._float_field(self._loss_target_edit, "loss target")
        factor      = self._float_field(self._factor_edit, "plateau factor")
        if None in (lr, lr_min, loss_target, factor):
            return None

        schedule = self._schedule_value()
        cmd = [
            "-u", TRAINING_SCRIPT,
            "--data",          *data_paths,
            "--output",        output_path,
            "--neighbourhood", nb_str,
            "--layers",        layers_str,
            "--out-size",      str(out_size),
            "--out-act",       out_act,
            "--epochs",        str(self._epochs_spin.value()),
            "--batch",         str(self._batch_spin.value()),
            "--lr",            str(lr),
            "--lr-schedule",   schedule,
            "--loss-target",   str(loss_target),
            "--print-every",   str(self._print_spin.value()),
        ]
        if schedule != "none":
            cmd += ["--lr-min", str(lr_min)]
        if schedule == "plateau":
            cmd += ["--plateau-patience", str(self._patience_spin.value()),
                    "--plateau-factor", str(factor)]

        mat_key = self._mat_key_edit.text().strip()
        if mat_key:
            cmd += ["--mat-key", mat_key]

        seed_text = self._seed_edit.text().strip()
        if seed_text:
            try:
                seed_val = int(seed_text)
            except ValueError:
                self._log(f"Error: '{seed_text}' is not a valid seed (must be an integer)\n")
                return None
            cmd += ["--seed", str(seed_val)]

        if self._open_radio.isChecked():
            cmd += ["--open-boundary"]

        if self._one_hot_radio.isChecked():
            states = self._one_hot_states_spin.value()
            cmd += ["--one-hot"] + ([str(states)] if states > 1 else [])

        if self._loss_csv_check.isChecked():
            suffix   = self._loss_csv_suffix_edit.text().strip()
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
