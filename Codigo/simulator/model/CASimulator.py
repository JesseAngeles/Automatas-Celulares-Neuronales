import os
import sys

core_path = os.path.join(os.path.dirname(__file__), "..", "core", "build")
sys.path.append(core_path)

import CASimulatorCore 
from model.CAConfig import CAConfig

class CASimulator:
    def __init__(self, config: CAConfig):
        self.core = CASimulatorCore.CASimulator(config.grid_size, config.neighbourhood,config.num_states, config.rule)

    def step(self):
        self.core.step()

    def print_grid(self):
        self.core.printGrid()

    def print_neighbourhood(self):
        self.core.printNeighbourhood()

    def print_neighbourhood(self):
        self.core.printNeighbourhood()

    def print_info(self):
        self.core.printInfo()

    def get_grid(self):
        return self.core.getGrid()

    def set_grid(self, grid):
        self.core.setGrid(grid)

    def set_rule(self, rule):
        self.core.setRule(rule)

    def set_neighbourhood(self, neighbourhood):
        self.core.setNeighbourhood(neighbourhood)

    def set_states(self, states):
        self.core.setStates(states)