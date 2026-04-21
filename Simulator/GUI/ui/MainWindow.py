# ui.MainWindow

from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import QMainWindow, QWidget, QHBoxLayout

from ui.ControlPane import ControlPane
from ui.SimulationWidget import SimulationWidget
from ui.ConfigPane import ConfigPane

from controllers.SimulationController import SimulationController
from controllers.SimulationConfig import SimulationConfig


APP_STYLESHEET = """
QMainWindow, QWidget#root {
    background-color: #f3f3f5;
}

QWidget {
    font-family: "Inter", "Segoe UI", "Helvetica Neue", sans-serif;
    font-size: 12px;
    color: #2a2a2e;
}

QWidget#panel {
    background-color: #ffffff;
    border: 1px solid #e3e3e8;
    border-radius: 10px;
}

QLabel#sectionTitle {
    color: #8a8a92;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1px;
}

QLabel#paneTitle {
    color: #1a1a1e;
    font-size: 14px;
    font-weight: 600;
}

QLabel#stepCounter {
    color: #1a1a1e;
    font-size: 18px;
    font-weight: 600;
}

QLabel#mutedValue {
    color: #8a8a92;
    font-size: 11px;
}

QFrame#sep {
    background: #ececf0;
    max-height: 1px;
    min-height: 1px;
    border: none;
}

QPushButton {
    background-color: #ffffff;
    color: #2a2a2e;
    border: 1px solid #d7d7dc;
    border-radius: 6px;
    padding: 6px 10px;
    min-height: 20px;
}
QPushButton:hover {
    background-color: #f5f5f8;
    border-color: #b8b8bf;
}
QPushButton:pressed {
    background-color: #eaeaef;
}
QPushButton:disabled {
    color: #b4b4ba;
    background-color: #fafafb;
    border-color: #e6e6ea;
}

QPushButton#primary {
    background-color: #3478f6;
    color: #ffffff;
    border: 1px solid #2f6de0;
    font-weight: 600;
}
QPushButton#primary:hover {
    background-color: #2f6de0;
}
QPushButton#primary:pressed {
    background-color: #2960c9;
}

QPushButton#ghost {
    background-color: transparent;
    border: 1px solid #e0e0e5;
    color: #4a4a50;
}
QPushButton#ghost:hover {
    background-color: #f5f5f8;
    border-color: #c8c8cf;
}

QSpinBox {
    background: #ffffff;
    border: 1px solid #d7d7dc;
    border-radius: 6px;
    padding: 3px 6px;
    min-height: 20px;
}
QSpinBox:focus {
    border-color: #3478f6;
}

QSlider::groove:horizontal {
    height: 4px;
    background: #e0e0e5;
    border-radius: 2px;
}
QSlider::sub-page:horizontal {
    background: #3478f6;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #ffffff;
    border: 1px solid #b8b8bf;
    width: 14px;
    height: 14px;
    margin: -6px 0;
    border-radius: 8px;
}
QSlider::handle:horizontal:hover {
    border-color: #3478f6;
}

QScrollArea {
    border: none;
    background: transparent;
}
QScrollBar:vertical {
    background: transparent;
    width: 8px;
    margin: 2px;
}
QScrollBar::handle:vertical {
    background: #d0d0d6;
    border-radius: 4px;
    min-height: 20px;
}
QScrollBar::handle:vertical:hover {
    background: #b0b0b8;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
"""


class MainWindow(QMainWindow):
    def __init__(self, grid_size):
        super().__init__()

        self.setWindowTitle("Cellular Automata Simulator")
        self.setFixedSize(QSize(1800, 1000))
        self.setStyleSheet(APP_STYLESHEET)

        self.config = SimulationConfig()

        self.control_pane = ControlPane()
        self.control_pane.setFixedWidth(260)

        self.simulation = SimulationWidget(grid_size=grid_size)

        self.config_pane = ConfigPane()
        self.config_pane.setFixedWidth(280)

        self.control_pane.reset_clicked.connect(self.handle_reset)
        self.control_pane.speed_moved.connect(self.handle_speed)

        self.controller = SimulationController(self.simulation, self.config)
        self.controller._on_running_changed = self.control_pane.update_play_button
        self.controller._on_step_changed = self.control_pane.update_step_label

        self.control_pane.connect_controller(self.controller)
        self.config_pane.connect_config(self.config)

        self.config_pane.neighbourhood_applied.connect(self.controller.on_neighbourhood_change)
        self.config_pane.neighbourhood_applied.connect(self._on_neighbourhood_applied)

        self.config_pane.states_applied.connect(self.controller.on_states_change)
        self.config_pane.states_applied.connect(
            lambda n, colors: self.simulation.set_state_colors(colors)
        )
        self.config_pane.states_applied.connect(self._on_states_applied)

        self.config_pane.active_state_changed.connect(self.simulation.set_active_state)
        self.config_pane.state_colors_changed.connect(self.simulation.set_state_colors)

        self.config_pane.rule_applied.connect(self.controller.on_rule_change)
        self.config_pane.rule_applied.connect(self._on_rule_number_applied)

        self.config_pane.rule_array_applied.connect(self.controller.on_rule_array_change)

        self.config_pane.boundary_changed.connect(self.controller.on_boundary_change)

        # Sincronizar estado inicial (GoL ya cargado en el controller)
        ca = self.controller.automaton.ca
        self.config_pane.set_rule_max(ca.get_r_size(), ca.get_states())
        self._update_rule_table()

        layout = QHBoxLayout()
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)
        layout.addWidget(self.control_pane)
        layout.addWidget(self.simulation, stretch=1)
        layout.addWidget(self.config_pane)

        container = QWidget()
        container.setObjectName("root")
        container.setLayout(layout)
        self.setCentralWidget(container)

    # ------------------------------------------------------------------

    def handle_reset(self):
        self.controller.on_erase_button()

    def handle_speed(self, new_speed):
        self.controller.on_speed_change(new_speed)

    def _update_rule_table(self):
        am = self.controller.automaton
        self.config_pane.sync_rule_table(
            group_sizes=list(am.getGroupSizes()),
            states=am.getStates(),
            rule=list(am.getRule()),
        )

    def _on_neighbourhood_applied(self, _):
        ca = self.controller.automaton.ca
        self.config_pane.set_rule_max(ca.get_r_size(), ca.get_states())
        self._update_rule_table()

    def _on_states_applied(self, _, __):
        ca = self.controller.automaton.ca
        self.config_pane.set_rule_max(ca.get_r_size(), ca.get_states())
        self._update_rule_table()

    def _on_rule_number_applied(self, _):
        self._update_rule_table()
