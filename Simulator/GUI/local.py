# local

from models.neuralCellularAutomaton import neuralCellularAutomaton as NCA

if __name__ == "__main__":
    size = 10
    space = [[0 for _ in range(size)] for _ in range(size)]
    
    neighbourhood = [[-1, -1], [-1,0], [-1, 1], 
                     [0, -1], [0,0], [0, 1], 
                     [1, -1], [1,0], [1, 1]]
    
    ca = NCA(space, neighbourhood)

    ca.print()