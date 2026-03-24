import random

GRID_SIZE = 5
MAX_STEPS = 50
ACTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]

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

    def move(self, action, is_predator=True):
        move = ACTIONS[action]
        if is_predator:
            self.predator[0] = max(0, min(GRID_SIZE-1, self.predator[0] + move[0]))
            self.predator[1] = max(0, min(GRID_SIZE-1, self.predator[1] + move[1]))
        else:
            self.prey[0] = max(0, min(GRID_SIZE-1, self.prey[0] + move[0]))
            self.prey[1] = max(0, min(GRID_SIZE-1, self.prey[1] + move[1]))

    def step(self, action):
        self.move(action, is_predator=True)   # Predator moves
        self.steps += 1

        # Temporary: Prey moves randomly (70% chance) - will be replaced later with learning prey
        if random.random() < 0.7:
            prey_action = random.randint(0, 3)
            self.move(prey_action, is_predator=False)

        # Terminal conditions
        if self.predator == self.prey:
            reward = 20
            done = True
        elif self.steps >= MAX_STEPS:
            reward = -20
            done = True
        else:
            reward = -1
            done = False

        return self.get_state(), reward, done