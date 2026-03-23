import numpy as np
import random

class GridWorld:
    def __init__(self, size=5):
        self.size = size
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)

        self.actions = ["up", "down", "left", "right"]
        self.obstacles = [(1,1), (2,2), (3,1)]
        self.reset()

    def reset(self):
        self.agent_pos = self.start
        return self.agent_pos

    def step(self, action):
        x, y = self.agent_pos

        # Add stochastic behavior
        if random.random() < 0.8:
            chosen_action = action  # 80% correct
        else:
            chosen_action = random.choice(self.actions)  # 20% random

        # Apply movement
        if chosen_action == "up":
            new_x = max(0, x - 1)
            new_y = y
        elif chosen_action == "down":
            new_x = min(self.size - 1, x + 1)
            new_y = y
        elif chosen_action == "left":
            new_x = x
            new_y = max(0, y - 1)
        elif chosen_action == "right":
            new_x = x
            new_y = min(self.size - 1, y + 1)

        # Obstacle check
        if (new_x, new_y) in self.obstacles:
            reward = -10
            new_x, new_y = x, y
        else:
            reward = -1

        self.agent_pos = (new_x, new_y)

        done = False
        if self.agent_pos == self.goal:
            reward = 10
            done = True

        return self.agent_pos, reward, done

    def get_state_space(self):
        return self.size * self.size

    def get_action_space(self):
        return len(self.actions)
    
    def render(self):
        grid = [["." for _ in range(self.size)] for _ in range(self.size)]

        for (i, j) in self.obstacles:
            grid[i][j] = "X"

        x, y = self.agent_pos
        gx, gy = self.goal

        grid[gx][gy] = "G"
        grid[x][y] = "A"

        for row in grid:
            print(" ".join(row))
        print("\n")