import random
import numpy as np
import pickle
import os

# Model save path
MODEL_DIR = "backend/agents/models"
MODEL_PATH = os.path.join(MODEL_DIR, "q_table.pkl")

class QLearningAgent:
    def __init__(self):
        self.q_table = {}
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05

        # Automatically load saved model if exists
        self.load_q_table()

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

    def save_q_table(self):
        """Save Q-table to disk"""
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(self.q_table, f)
        print(f"✅ Q-table saved! ({len(self.q_table)} states learned)")

    def load_q_table(self):
        """Load Q-table from disk if it exists"""
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, "rb") as f:
                self.q_table = pickle.load(f)
            print(f"✅ Loaded existing Q-table with {len(self.q_table)} states")
        else:
            print("ℹ️ No saved Q-table found. Starting fresh.")