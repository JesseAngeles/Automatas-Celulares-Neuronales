# controllers.SimulationController

from PyQt5.QtCore import QTimer
from models.ca import CA

class SimulationController:
    def __init__(self, simulation_widget, config):
        self.simulation_widget = simulation_widget

        self.config = config

        self.automaton = CA(
            simulation_widget.grid_size,
            simulation_widget.grid_size,
            [[0] * simulation_widget.grid_size for _ in range(simulation_widget.grid_size)]
            )
        
        # Timer config
        self.timer = QTimer()
        self.timer .timeout.connect(self.on_step_button)
        self.is_running = False

        self.interval = 200
        self.timer.setInterval(self.interval)

        # Evolution records
        self.evolution: list = []

    # Button functions
    def on_increase_speed_button(self):
        self.interval = max(10, self.interval - 50)
        self.timer.setInterval(self.interval)

    def on_decrease_speed_button(self):
        self.interval += 50
        self.timer.setInterval(self.interval)

    def on_step_button(self):
        # Python to C++
        self.automaton.set_space(self.simulation_widget.grid)
        self.evolution.append(self.automaton.get_space())

        # ejecutar regla
        self.automaton.step()

        # C++ to Python
        new_grid = self.automaton.get_space()
        self.simulation_widget.update_grid(new_grid)
        
    def on_play_button(self):
        if self.is_running:
            self.timer.stop()
            self.is_running = False
        else:
            self.timer.start()
            self.is_running = True

    def on_backstep_button(self):
        if self.evolution:
            new_grid = self.evolution.pop()
            self.simulation_widget.update_grid(new_grid)    
            self.automaton.set_space(new_grid)

    def on_erase_button(self):
        new_grid = [[0] * self.simulation_widget.grid_size for _ in range(self.simulation_widget.grid_size)]
    
        self.simulation_widget.update_grid(new_grid)
        self.automaton.set_space(new_grid)

        if self.is_running:
            self.on_play_button()