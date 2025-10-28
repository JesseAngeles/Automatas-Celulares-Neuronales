moore = [[-1,-1],[-1,0],[-1,1],
         [0,-1],[0,0],[0,1],
         [1,-1],[1,0],[1,1]]

class CAConfig:
    def __init__(self, num_states = 2, neighbourhood = moore, frontier = None, rule=10):
        self.num_states = num_states
        self.neighbourhood = neighbourhood
        self.frontier = frontier
        self.rule = 10