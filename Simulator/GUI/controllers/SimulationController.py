# controllers.SimulationController

from PyQt5.QtCore import QTimer
from models.cellularAutomaton import cellularAutomaton
import random

class SimulationController:
    def __init__(self, simulation_widget, config):
        self.simulation_widget = simulation_widget

        self.config = config

        self.automaton = cellularAutomaton(
            space = [[0] * simulation_widget.grid_size for _ in range(simulation_widget.grid_size)],
            states = 2,
            neighbourhood = [[-1,-1],[0,-1],[1,-1]],
            rule = [0,1,0,1,1,0,1,0] 
            )
        
        # Timer config
        self.timer = QTimer()
        self.timer .timeout.connect(self.on_step_button)
        self.is_running = False

        self.interval = 200
        self.timer.setInterval(self.interval)

        # Evolution records
        self.evolution: list = []

        # Event click
        self.simulation_widget.cell_toggled.connect(self.on_cell_toggled)

    # Mouse events
    def on_cell_toggled(self, x, y, new_val):
        self.automaton.setCell(x, y, new_val)
        self.evolution.clear()

    # Buttons events
    def on_increase_speed_button(self):
        self.interval = max(10, self.interval - 50)
        self.timer.setInterval(self.interval)

    def on_decrease_speed_button(self):
        self.interval += 50
        self.timer.setInterval(self.interval)

    def on_step_button(self):
        # Python to C++
        self.evolution.append(self.automaton.getSpace().copy())

        # ejecutar regla
        self.automaton.step()

        # C++ to Python
        self.simulation_widget.update_grid(self.automaton.getSpace())
    
    def on_play_button(self):
        if self.is_running:
            self.timer.stop()
            self.is_running = False
        else:
            self.timer.start()
            self.is_running = True

    def on_backstep_button(self):
        print(self.evolution)
        if self.evolution:
            new_grid = self.evolution.pop()
            self.simulation_widget.update_grid(new_grid)    
            self.automaton.setSpace(new_grid)

    def on_erase_button(self):
        new_grid = [[0] * self.simulation_widget.grid_size for _ in range(self.simulation_widget.grid_size)]
    
        self.simulation_widget.update_grid(new_grid)
        self.automaton.setSpace(new_grid)

        if self.is_running:
            self.on_play_button()

    def on_random_button(self):
        states = self.automaton.getStates()

        new_grid = [
            [random.randint(0, states - 1) for _ in range(self.simulation_widget.grid_size)]
            for _ in range(self.simulation_widget.grid_size)
        ]

        self.simulation_widget.update_grid(new_grid)
        self.automaton.setSpace(new_grid)

        if self.is_running:
            self.on_play_button()