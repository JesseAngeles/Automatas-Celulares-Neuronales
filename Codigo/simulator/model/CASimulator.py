from model.CAConfig import CAConfig

class CASimulator:
    def __init__(self):
        self.num_states = 0

    def set_config(self, config: CAConfig):
        self.num_states = config.num_states

    def step(self):
        pass

    def reset(self):
        pass