# models.ca.py

import core 

class CA:
    def __init__(self, width: int, height: int, space: list):
        self.ca = core.Life(width, height, space);

    def step(self):
        self.ca.step()

    def print(self):
        self.ca.print()

    # Getters
    def get_space(self):
        return self.ca.get_space()

    def get_width(self):
        return self.ca.get_width()

    def get_height(self):
        return self.ca.get_height()

    def get_cell(self, x: int, y: int):
        return self.ca.get_cell(x, y)

    # Setters
    def set_space(self, space: list):
        self.ca.set_space(space)

    def set_cell(self, x: int, y: int, value: list):
        self.ca.set_cell(x, y, value)