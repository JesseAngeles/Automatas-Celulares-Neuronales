life = [
    [[-1, -1], [-1, 0], [-1, 1],
     [0, -1], [0, 1],
     [1, -1], [1, 0], [1, 1]],
    [[0, 0]]
]

class CAConfig:
    def __init__(self, num_states=2, neighbourhood=life,
                 frontier=None, rule=6152, grid_size=21,
                 cell_size=40, grid_neighbour_size=9,
                 cell_neighbour_size=50):
        
        self.num_states = num_states
        self.neighbourhood = neighbourhood
        self.frontier = frontier
        self.rule = rule
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.grid_neighbour_size = grid_neighbour_size
        self.cell_neighbour_size = cell_neighbour_size

        self.evolution: list