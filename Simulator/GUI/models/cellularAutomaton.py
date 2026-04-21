# models.cellularAutomaton.py

from CA import CA
import numpy as np

class cellularAutomaton:

    # Constructor
    # neighbourhood: lista de grupos, cada grupo es lista de [dx, dy]
    # Ej: [[[-1,0],[1,0]], [[0,0]]]  → grupo vecinos + grupo celda central
    def __init__(self, space: list[list[int]], states: int, neighbourhood: list[list[list[int]]], rule: list[int]):
        height = len(space)
        width = len(space[0])

        flat, group_sizes = self._parse_neighbourhood(neighbourhood)

        new_space      = np.array(space, dtype=np.int32).flatten()
        new_neighbourhood = np.array(flat, dtype=np.int32)
        new_group_sizes   = np.array(group_sizes, dtype=np.int32)
        new_rule          = np.array(rule, dtype=np.int32)

        n_size   = len(flat)
        n_groups = len(group_sizes)
        r_size   = len(rule)

        self.ca = CA(
            width, height,
            new_space,
            states,
            new_neighbourhood, n_size,
            new_group_sizes, n_groups,
            new_rule, r_size
        )

    # Calcula r_size para una vecindad agrupada dada
    @staticmethod
    def ruleSize(states: int, neighbourhood: list[list[list[int]]]) -> int:
        r = 1
        for group in neighbourhood:
            r *= (states - 1) * len(group) + 1
        return r

    @staticmethod
    def _parse_neighbourhood(neighbourhood):
        flat = []
        group_sizes = []
        for group in neighbourhood:
            flat.extend(group)
            group_sizes.append(len(group))
        return flat, group_sizes

    def step(self):
        self.ca.step()

    def run(self, steps: int):
        return self.ca.run(steps)

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

    def getGroupSizes(self):
        return self.ca.get_group_sizes()

    def getNGroups(self):
        return self.ca.get_n_groups()

    def getRule(self):
        return self.ca.get_rule()

    def getCell(self, x: int, y: int):
        return self.ca.get_cell(x, y)

    # Setters
    def setSpace(self, space):
        self.ca.set_space(np.array(space, dtype=np.int32))

    def setStates(self, states: int):
        self.ca.set_states(states)

    def setNeighbourhood(self, neighbourhood: list[list[list[int]]]):
        flat, group_sizes = self._parse_neighbourhood(neighbourhood)
        self.ca.set_neighbourhood(
            np.array(flat, dtype=np.int32),
            np.array(group_sizes, dtype=np.int32)
        )

    def setRule(self, rule):
        self.ca.set_rule(np.array(rule, dtype=np.int32))

    def setCell(self, x: int, y: int, value: int):
        self.ca.set_cell(x, y, value)

    def setBoundary(self, mode: int):
        self.ca.set_boundary(mode)

    def getBoundary(self) -> int:
        return self.ca.get_boundary()
