import random

GRID_SIZE = 5
ACTIONS = [(1,0), (-1,0), (0,1), (0,-1)]

class PredatorPreyEnv:
    def __init__(self):
        self.steps = 0
        self.episode = 1
        self.reset()

    def reset(self):
        self.predator = [0, 0]
        self.prey = [4, 4]
        self.steps = 0
        return self.get_state()

    def get_state(self):
        return tuple(self.predator + self.prey)

    def move(self, action):
        move = ACTIONS[action]

        self.predator[0] = max(0, min(GRID_SIZE-1, self.predator[0] + move[0]))
        self.predator[1] = max(0, min(GRID_SIZE-1, self.predator[1] + move[1]))

    def step(self, action):
        self.move(action)
        self.steps += 1

        if self.predator == self.prey:
            return self.get_state(), 10, True
        else:
            return self.get_state(), -1, False