import random
import numpy as np

class QLearningAgent:
    def __init__(self):
        self.q_table = {}
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.2

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        return int(np.argmax(self.q_table.get(state, [0]*4)))

    def update(self, state, action, reward, next_state):
        self.q_table.setdefault(state, [0]*4)
        self.q_table.setdefault(next_state, [0]*4)

        best_next = max(self.q_table[next_state])

        self.q_table[state][action] += self.alpha * (
            reward + self.gamma * best_next - self.q_table[state][action]
        )