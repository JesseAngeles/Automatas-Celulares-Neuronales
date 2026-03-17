# models.cellularAutomaton.py

import CA # type: ignore
import numpy as np

class cellularAutomaton:

    # Constructor
    def __init__(self, space: list[list[int]], states: int, neighbourhood: list[list[int]], rule: list[int]):
        height = len(space)
        width = len(space[0])

        new_space = np.array(space, dtype=np.int32).flatten()
        new_neighbourhood = np.array(neighbourhood, dtype=np.int32).flatten()
        new_rule = np.array(rule, dtype=np.int32)

        self.ca = CA.CA(
            width,
            height,
            new_space,
            states,
            new_neighbourhood,
            len(neighbourhood),
            new_rule,
            len(rule)            
        )

    def step(self):
        self.ca.step()

    def print(self):
        self.ca.print()

    # Getters

    def getSpace(self):
        return self.ca.get_space()
    
    def getWidth(self):
        return self.ca.get_width()
    
    def getHeight(self):
        return self.ca.get_height()
    
    def getStates(self):
        return self.ca.get_states()

    def getNeighbourhood(self):
        return self.ca.get_neighbourhood()

    def getRule(self):
        return self.ca.get_rule()

    def getCell(self, x: int, y: int):
        return self.ca.get_cell(x, y)
    
    # Setters
    def setSpace(self, space):
        self.ca.set_space(space)

    def setStates(self, states:int):
        self.ca.set_states(states) 

    def setNeighbourhood(self, neighbourhood):
        self.ca.set_neighbourhood(neighbourhood)

    def setRule(self, rule):
        self.ca.set_rule(rule)

    def setCell(self, x: int, y: int, value: int):
        self.ca.set_cell(x, y, value)