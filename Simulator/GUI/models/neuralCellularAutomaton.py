# models.neuralCellularAutomaton

from CA import NCA
import numpy as np

class neuralCellularAutomaton:
    # Constructor
    def __init__(self, space: list[list[float]], neighbourhood: list[list[int]]):
        height = len(space)
        width = len(space[0])

        new_space = np.array(space, dtype = np.float32).flatten()

        new_neighbourhood = np.array(neighbourhood, dtype=np.int32)
        self.nca = NCA(width, height, new_space, new_neighbourhood, len(neighbourhood))

    def step(self):
        self.nca.step()

    def print(self):
        self.nca.print()

    def set_weights(self, weights, biases, layer_in, layer_out, activations):
        self.nca.set_weights(weights, biases, layer_in, layer_out, activations)

    # Getters
    def get_width(self):
        return self.nca.get_width()
    
    def get_height(self):
        return self.nca.get_height()
    
    def get_space(self):
        return self.nca.get_space()
    
    def get_cell(self, x: int, y: int):
        return self.nca.get_cell(x, y)
    
    def get_neighbourhood(self):
        return self.nca.get_neighbourhood()

    def get_n_size(self):
        return self.nca.get_n_size()
    
    # Setters
    def setSpace(self, space):
        self.nca.set_space(space)

    def setCell(self, x: int, y: int, value: int):
        self.nca.set_cell(x, y, value)

    def setNeighbourhood(self, neighbourhood):
        self.nca.set_neighbourhood(neighbourhood)