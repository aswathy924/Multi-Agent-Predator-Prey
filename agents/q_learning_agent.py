import numpy as np
import random

class QLearningAgent:
    def __init__(self, state_size, action_size, lr=0.1, gamma=0.9, epsilon=1.0):
        self.state_size = state_size
        self.action_size = action_size

        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

        self.q_table = np.zeros((state_size, action_size))

    def state_to_index(self, state, grid_size):
        px, py, qx, qy = state

        return (
            px * grid_size**3 +
            py * grid_size**2 +
            qx * grid_size +
            qy
        )

    def choose_action(self, state_idx):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_size - 1)
        return np.argmax(self.q_table[state_idx])

    def update(self, state_idx, action, reward, next_state_idx):
        best_next = np.max(self.q_table[next_state_idx])

        self.q_table[state_idx, action] = (
            self.q_table[state_idx, action]
            + self.lr * (reward + self.gamma * best_next - self.q_table[state_idx, action])
        )