# ui.ConfigPane

import random
from PyQt5.QtWidgets import (QWidget, QPushButton, QVBoxLayout, QHBoxLayout,
                              QLabel, QSpinBox, QFrame, QScrollArea, QColorDialog)
from PyQt5.QtGui import QColor
from PyQt5.QtCore import pyqtSignal, Qt

from ui.NeighbourWidget import NeighbourWidget

DEFAULT_STATE_COLORS = [QColor(255, 255, 255), QColor(0, 0, 0)]

BOUNDARY_LABELS = ["Open", "Periodic", "Reflective", "Reflexive"]

LIFE_NEIGHBOURHOOD = [
    [[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]],
    [[0,0]]
]
LIFE_GROUP_COLORS = [QColor(50, 120, 220), QColor(210, 120, 40)]


def _rand_color() -> QColor:
    return QColor(random.randint(60, 210), random.randint(60, 210), random.randint(60, 210))


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


class ConfigPane(QWidget):

    neighbourhood_applied = pyqtSignal(object)      # list[list[list[int]]]
    states_applied        = pyqtSignal(int, object) # (n_states, list[QColor])
    active_state_changed  = pyqtSignal(int)
    state_colors_changed  = pyqtSignal(object)      # list[QColor]
    rule_applied          = pyqtSignal(int)
    rule_array_applied    = pyqtSignal(object)      # list[int]
    boundary_changed      = pyqtSignal(int)         # 0=Open,1=Periodic,2=Reflective,3=Reflexive

    def __init__(self):
        super().__init__()
        self.setObjectName("panel")

        self.neighbout_widget = NeighbourWidget()

        # ── Título del panel ──────────────────────────────────────────
        self.pane_title = QLabel("Configuration")
        self.pane_title.setObjectName("paneTitle")

        # ── Section labels ────────────────────────────────────────────
        self.neighbour_label = _section_label("Neighbourhood")
        self.states_label    = _section_label("States")
        self.frontier_label  = _section_label("Frontier")
        self.rule_label      = _section_label("Rule")

        # ── Buttons ───────────────────────────────────────────────────
        self._boundary_mode  = 1  # PERIODIC — igual que el default del core
        self.frontier_button = QPushButton(BOUNDARY_LABELS[self._boundary_mode])
        self.frontier_button.clicked.connect(self._on_frontier_button)
        self.states_button   = QPushButton("Apply")
        self.rule_button     = QPushButton("Apply")
        self.states_button.setObjectName("ghost")
        self.rule_button.setObjectName("ghost")

        # ── Inputs ────────────────────────────────────────────────────
        self.states_spin = QSpinBox()
        self.states_spin.setRange(2, 20)
        self.states_spin.setValue(2)

        self.rule_spin = QSpinBox()
        self.rule_spin.setRange(0, 10000)
        self.rule_spin.setValue(0)

        # ── Panel de grupos (derecha del grid) ────────────────────────
        self._active_group = 1

        self._groups_container = QWidget()
        self._groups_layout    = QVBoxLayout(self._groups_container)
        self._groups_layout.setContentsMargins(0, 0, 0, 0)
        self._groups_layout.setSpacing(2)

        groups_scroll = QScrollArea()
        groups_scroll.setWidgetResizable(True)
        groups_scroll.setWidget(self._groups_container)
        groups_scroll.setFrameShape(QFrame.NoFrame)
        groups_scroll.setFixedHeight(self.neighbout_widget.height())
        groups_scroll.setMinimumWidth(90)

        self._add_btn = QPushButton("+ Add")
        self._del_btn = QPushButton("- Del")
        self._add_btn.clicked.connect(self._on_add_group)
        self._del_btn.clicked.connect(self._on_remove_group)

        group_btn_row = QHBoxLayout()
        group_btn_row.addWidget(self._add_btn)
        group_btn_row.addWidget(self._del_btn)

        groups_side = QVBoxLayout()
        groups_side.setSpacing(4)
        groups_side.addWidget(groups_scroll)
        groups_side.addLayout(group_btn_row)

        nb_row = QHBoxLayout()
        nb_row.setSpacing(8)
        nb_row.addWidget(self.neighbout_widget)
        nb_row.addLayout(groups_side)

        self._apply_btn = QPushButton("Apply")
        self._apply_btn.clicked.connect(self._on_apply_neighbourhood)

        # ── Lista de estados (pintura) ─────────────────────────────────
        self._state_colors: list[QColor] = list(DEFAULT_STATE_COLORS)
        self._active_paint_state = 1

        self._states_container = QWidget()
        self._states_layout    = QVBoxLayout(self._states_container)
        self._states_layout.setContentsMargins(0, 0, 0, 0)
        self._states_layout.setSpacing(4)

        states_scroll = QScrollArea()
        states_scroll.setWidgetResizable(True)
        states_scroll.setWidget(self._states_container)
        states_scroll.setMaximumHeight(120)
        states_scroll.setFrameShape(QFrame.NoFrame)

        # ── Tabla de regla ─────────────────────────────────────────────
        self._rule_array:       list[int] = []
        self._rule_states:      int       = 2
        self._rule_group_sizes: list[int] = []

        self._rule_table_container = QWidget()
        self._rule_table_layout    = QVBoxLayout(self._rule_table_container)
        self._rule_table_layout.setContentsMargins(2, 2, 2, 2)
        self._rule_table_layout.setSpacing(1)

        rule_table_scroll = QScrollArea()
        rule_table_scroll.setWidgetResizable(True)
        rule_table_scroll.setWidget(self._rule_table_container)
        rule_table_scroll.setFrameShape(QFrame.Box)

        # ── Layout principal ──────────────────────────────────────────
        layout = QVBoxLayout()
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(8)

        states_layout = QHBoxLayout()
        states_layout.setSpacing(6)
        rule_layout   = QHBoxLayout()
        rule_layout.setSpacing(6)

        layout.addWidget(self.pane_title)
        layout.addWidget(_separator())

        layout.addWidget(self.neighbour_label)
        layout.addLayout(nb_row)
        layout.addWidget(self._apply_btn)
        layout.addWidget(_separator())

        layout.addWidget(self.states_label)
        states_layout.addWidget(self.states_spin)
        states_layout.addWidget(self.states_button)
        layout.addLayout(states_layout)
        layout.addWidget(states_scroll)
        layout.addWidget(_separator())

        layout.addWidget(self.frontier_label)
        layout.addWidget(self.frontier_button)
        layout.addWidget(_separator())

        layout.addWidget(self.rule_label)
        rule_layout.addWidget(self.rule_spin)
        rule_layout.addWidget(self.rule_button)
        layout.addLayout(rule_layout)
        layout.addWidget(rule_table_scroll, stretch=1)

        self.setLayout(layout)

        # ── Conexiones internas ───────────────────────────────────────
        self.states_button.clicked.connect(self._on_states_button)
        self.rule_button.clicked.connect(self._on_rule_button)

        # ── Valores iniciales: Life ───────────────────────────────────
        self.neighbout_widget.group_colors[1] = LIFE_GROUP_COLORS[0]
        self.neighbout_widget.add_group(LIFE_GROUP_COLORS[1])
        self.neighbout_widget.load_neighbourhood(LIFE_NEIGHBOURHOOD)
        self._rebuild_groups_panel()
        self._rebuild_states_panel()

    # ------------------------------------------------------------------
    # Color pickers
    # ------------------------------------------------------------------
    def _pick_state_color(self, i: int):
        color = QColorDialog.getColor(self._state_colors[i], self)
        if color.isValid():
            self._state_colors[i] = color
            self._rebuild_states_panel()
            self._rebuild_rule_table()
            self.state_colors_changed.emit(list(self._state_colors))

    def _pick_group_color(self, state: int):
        current = self.neighbout_widget.group_colors[state]
        color = QColorDialog.getColor(current, self)
        if color.isValid():
            self.neighbout_widget.group_colors[state] = color
            self.neighbout_widget.update()
            self._rebuild_groups_panel()

    # ------------------------------------------------------------------
    # Tabla de regla
    # ------------------------------------------------------------------
    def sync_rule_table(self, group_sizes: list, states: int, rule: list):
        self._rule_group_sizes = list(group_sizes)
        self._rule_states      = states
        self._rule_array       = list(rule)
        self._rebuild_rule_table()
        self._sync_spinner_from_array()

    def _sync_spinner_from_array(self):
        k = self._rule_states
        n = sum(self._rule_array[i] * (k ** i) for i in range(len(self._rule_array)))
        if n <= 2_147_483_647:
            self.rule_spin.setValue(n)

    def _decode_rule_index(self, idx: int) -> list[int]:
        """Convierte un índice de regla (orden CUDA) a lista de sumas por grupo."""
        k      = self._rule_states
        ranges = [(k - 1) * g + 1 for g in self._rule_group_sizes]
        n      = len(self._rule_group_sizes)
        sums   = [0] * n
        for g in range(n - 1, -1, -1):   # último grupo = menos significativo
            sums[g] = idx % ranges[g]
            idx    //= ranges[g]
        return sums

    def _rebuild_rule_table(self):
        while self._rule_table_layout.count():
            item = self._rule_table_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not self._rule_group_sizes or not self._rule_array:
            self._rule_table_layout.addStretch()
            return

        n_groups = len(self._rule_group_sizes)

        # Cabecera
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(2, 1, 2, 1)
        pattern_header = QLabel("  ".join(f"G{g+1}" for g in range(n_groups)))
        pattern_header.setStyleSheet("font-size: 10px; font-weight: bold; font-family: monospace;")
        output_header = QLabel("Out")
        output_header.setStyleSheet("font-size: 10px; font-weight: bold;")
        output_header.setFixedWidth(36)
        header_layout.addWidget(pattern_header, stretch=1)
        header_layout.addWidget(output_header)
        header_widget = QWidget()
        header_widget.setLayout(header_layout)
        self._rule_table_layout.addWidget(header_widget)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #aaa;")
        self._rule_table_layout.addWidget(sep)

        # Filas
        for idx, out_state in enumerate(self._rule_array):
            sums = self._decode_rule_index(idx)

            pattern_text = "  ".join(f"{s:>{len(str((self._rule_states-1)*g))}}"
                                     for s, g in zip(sums, self._rule_group_sizes))

            pattern_lbl = QLabel(pattern_text)
            pattern_lbl.setStyleSheet("font-size: 10px; font-family: monospace;")

            color = (self._state_colors[out_state]
                     if out_state < len(self._state_colors)
                     else QColor(128, 128, 128))

            swatch = QFrame()
            swatch.setFixedSize(14, 14)
            swatch.setStyleSheet(f"background-color: {color.name()}; border: 1px solid #666;")

            num_lbl = QLabel(str(out_state))
            num_lbl.setStyleSheet("font-size: 10px;")
            num_lbl.setFixedWidth(14)

            out_layout = QHBoxLayout()
            out_layout.setContentsMargins(2, 0, 2, 0)
            out_layout.setSpacing(3)
            out_layout.addWidget(swatch)
            out_layout.addWidget(num_lbl)

            out_widget = QWidget()
            out_widget.setFixedWidth(36)
            out_widget.setLayout(out_layout)
            out_widget.setCursor(Qt.PointingHandCursor)
            out_widget.mousePressEvent = lambda _e, i=idx: self._cycle_rule_output(i)

            row_layout = QHBoxLayout()
            row_layout.setContentsMargins(2, 0, 2, 0)
            row_layout.setSpacing(4)
            row_layout.addWidget(pattern_lbl, stretch=1)
            row_layout.addWidget(out_widget)

            row_widget = QWidget()
            row_widget.setLayout(row_layout)
            row_widget.setFixedHeight(20)

            self._rule_table_layout.addWidget(row_widget)

        self._rule_table_layout.addStretch()

    def _cycle_rule_output(self, idx: int):
        self._rule_array[idx] = (self._rule_array[idx] + 1) % self._rule_states
        self._rebuild_rule_table()
        self._sync_spinner_from_array()
        self.rule_array_applied.emit(list(self._rule_array))

    # ------------------------------------------------------------------
    # Regla (número)
    # ------------------------------------------------------------------
    def _on_rule_button(self):
        self.rule_applied.emit(self.rule_spin.value())

    def set_rule_max(self, r_size: int, states: int = 2):
        self.rule_spin.setRange(0, min(states ** r_size - 1, 2_147_483_647))

    # ------------------------------------------------------------------
    # Panel de grupos (neighbourhood)
    # ------------------------------------------------------------------
    def _rebuild_groups_panel(self):
        while self._groups_layout.count():
            item = self._groups_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self._groups_layout.addWidget(
            self._make_group_row(0, QColor(255, 255, 255), "Erase")
        )
        for i in range(1, self.neighbout_widget.n_groups() + 1):
            color = self.neighbout_widget.group_colors[i]
            self._groups_layout.addWidget(
                self._make_group_row(i, color, f"Group {i}")
            )
        self._groups_layout.addStretch()

        self._del_btn.setEnabled(self.neighbout_widget.n_groups() > 1)

    def _make_group_row(self, state: int, color: QColor, label_text: str) -> QWidget:
        is_active = (state == self._active_group)
        border    = "2px solid #222" if is_active else "1px solid #888"

        swatch = QFrame()
        swatch.setFixedSize(16, 16)
        swatch.setStyleSheet(f"background-color: {color.name()}; border: {border};")

        label = QLabel(label_text)
        label.setStyleSheet("font-size: 11px;")

        row_layout = QHBoxLayout()
        row_layout.setSpacing(4)
        row_layout.setContentsMargins(2, 1, 2, 1)
        row_layout.addWidget(swatch)
        row_layout.addWidget(label)
        row_layout.addStretch()

        wrapper = QWidget()
        wrapper.setLayout(row_layout)
        wrapper.setCursor(Qt.PointingHandCursor)
        wrapper.mousePressEvent = lambda _e, s=state: self._select_group(s)
        if state > 0:
            wrapper.mouseDoubleClickEvent = lambda _e, s=state: self._pick_group_color(s)

        return wrapper

    def _select_group(self, state: int):
        self._active_group = state
        self.neighbout_widget.set_active_state(state)
        self._rebuild_groups_panel()

    def _on_add_group(self):
        new_state = self.neighbout_widget.add_group()
        self._select_group(new_state)

    def _on_remove_group(self):
        removed = self.neighbout_widget.remove_last_group()
        if removed:
            self._active_group = self.neighbout_widget.active_state
            self._rebuild_groups_panel()

    def _on_apply_neighbourhood(self):
        nb = self.neighbout_widget.get_neighbourhood()
        if nb:
            self.neighbourhood_applied.emit(nb)

    # ------------------------------------------------------------------
    # Panel de estados (pintura)
    # ------------------------------------------------------------------
    def _rebuild_states_panel(self):
        while self._states_layout.count():
            item = self._states_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for i, color in enumerate(self._state_colors):
            is_active = (i == self._active_paint_state)
            border    = "2px solid #222" if is_active else "1px solid #888"

            swatch = QFrame()
            swatch.setFixedSize(16, 16)
            swatch.setStyleSheet(f"background-color: {color.name()}; border: {border};")

            label = QLabel("Erase" if i == 0 else f"State {i}")
            label.setStyleSheet("font-size: 11px;")

            row_layout = QHBoxLayout()
            row_layout.setSpacing(6)
            row_layout.setContentsMargins(2, 1, 2, 1)
            row_layout.addWidget(swatch)
            row_layout.addWidget(label)
            row_layout.addStretch()

            wrapper = QWidget()
            wrapper.setLayout(row_layout)
            wrapper.setCursor(Qt.PointingHandCursor)
            wrapper.mousePressEvent       = lambda _e, s=i: self._select_paint_state(s)
            wrapper.mouseDoubleClickEvent = lambda _e, s=i: self._pick_state_color(s)

            self._states_layout.addWidget(wrapper)

    def _select_paint_state(self, state: int):
        self._active_paint_state = state
        self._rebuild_states_panel()
        self.active_state_changed.emit(state)

    def _on_states_button(self):
        new_n = self.states_spin.value()

        while len(self._state_colors) < new_n:
            self._state_colors.append(_rand_color())
        self._state_colors = self._state_colors[:new_n]

        if self._active_paint_state >= new_n:
            self._active_paint_state = 0
            self.active_state_changed.emit(0)

        self._rebuild_states_panel()
        self.states_applied.emit(new_n, list(self._state_colors))

    # ------------------------------------------------------------------
    # Frontera
    # ------------------------------------------------------------------
    def _on_frontier_button(self):
        self._boundary_mode = (self._boundary_mode + 1) % len(BOUNDARY_LABELS)
        self.frontier_button.setText(BOUNDARY_LABELS[self._boundary_mode])
        self.boundary_changed.emit(self._boundary_mode)

    # ------------------------------------------------------------------
    # Conexiones externas
    # ------------------------------------------------------------------
    def connect_config(self, config):
        pass
