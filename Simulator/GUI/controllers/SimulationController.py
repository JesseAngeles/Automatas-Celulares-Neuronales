# controllers.SimulationController

import random
from PyQt5.QtCore import QTimer
from collections import deque

from models.cellularAutomaton import cellularAutomaton

PREFETCH_SIZE = 10

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
        self.past_record: list = []
        self.future_record:deque = deque([])         

        # Event click
        self.simulation_widget.cell_toggled.connect(self.on_cell_toggled)

    def _clear_history(self):
        self.past_record.clear()
        self.future_record.clear()

    def _prefetch(self):
        if len(self.future_record) < PREFETCH_SIZE:
            self.future_record.extend(self.automaton.run(PREFETCH_SIZE))

    # Mouse events
    def on_cell_toggled(self, x, y, new_val):
        self.automaton.setSpace(self.simulation_widget.grid)
        self._clear_history()
        self.automaton.setCell(x, y, new_val)
        current = self.automaton.getSpace()
        self.simulation_widget.update_grid(current)

    # Slider events
    def on_speed_change(self, new_speed):
        self.interval = 1000 - new_speed
        self.timer.setInterval(self.interval) 

    # Buttons events
    def on_step_button(self):
        # Se añade al actual al pasado
        self.past_record.append(self.simulation_widget.grid.copy())
        self._prefetch()

        self.simulation_widget.update_grid(self.future_record.popleft().copy())

    def on_play_button(self):
        if self.is_running:
            self.timer.stop()
            self.is_running = False
        else:
            self.timer.start()
            self.is_running = True

    def on_backstep_button(self):
        if not self.past_record:
            return
        # El frame mostrado va al future

        self.future_record.appendleft(self.simulation_widget.grid.copy())
        self.simulation_widget.update_grid(self.past_record.pop())

    def on_erase_button(self):
        new_grid = [[0] * self.simulation_widget.grid_size for _ in range(self.simulation_widget.grid_size)]
        self._clear_history()                    # ← 
        self.automaton.setSpace(new_grid)
        self.simulation_widget.update_grid(new_grid)
        if self.is_running:
            self.on_play_button()

    def on_random_button(self):
        states = self.automaton.getStates()
        new_grid = [
            [random.randint(0, states - 1) for _ in range(self.simulation_widget.grid_size)]
            for _ in range(self.simulation_widget.grid_size)
        ]
        self._clear_history()                    # ←
        self.automaton.setSpace(new_grid)
        self.simulation_widget.update_grid(new_grid)
        if self.is_running:
            self.on_play_button()