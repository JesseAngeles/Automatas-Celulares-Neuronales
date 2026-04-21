# ui.MainWindow

from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import QMainWindow, QWidget, QHBoxLayout

from ui.ControlPane import ControlPane
from ui.SimulationWidget import SimulationWidget
from ui.ConfigPane import ConfigPane

from controllers.SimulationController import SimulationController
from controllers.SimulationConfig import SimulationConfig

class MainWindow(QMainWindow):
    def __init__(self, grid_size):
        super().__init__()

        self.setWindowTitle("Cellular Automata Simulator")
        self.setFixedSize(QSize(1800, 1000))

        self.config = SimulationConfig()

        self.control_pane = ControlPane()
        self.control_pane.setFixedWidth(250)

        self.simulation = SimulationWidget(grid_size=grid_size)

        self.config_pane = ConfigPane()
        self.config_pane.setFixedWidth(250)

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
        layout.addWidget(self.control_pane)
        layout.addWidget(self.simulation)
        layout.addWidget(self.config_pane)

        container = QWidget()
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
