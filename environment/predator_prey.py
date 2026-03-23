import random

class PredatorPreyEnv:
    def __init__(self, size=5):
        self.size = size
        self.actions = ["up", "down", "left", "right"]
        self.reset()

    def reset(self):
        # Predator starts at top-left
        self.predator_pos = (0, 0)

        # Prey starts randomly
        self.prey_pos = (self.size - 1, self.size - 1)

        return self.get_state()

    def get_state(self):
        # State = (predator_x, predator_y, prey_x, prey_y)
        return (*self.predator_pos, *self.prey_pos)

    def move(self, pos, action):
        x, y = pos

        if action == "up":
            x = max(0, x - 1)
        elif action == "down":
            x = min(self.size - 1, x + 1)
        elif action == "left":
            y = max(0, y - 1)
        elif action == "right":
            y = min(self.size - 1, y + 1)

        return (x, y)

    def step(self, action):
        # Predator moves
        new_predator_pos = self.move(self.predator_pos, action)

        # Check capture BEFORE prey moves
        if new_predator_pos == self.prey_pos:
            self.predator_pos = new_predator_pos
            return self.get_state(), 10, True

        # Update predator position
        self.predator_pos = new_predator_pos

        # Prey moves
        prey_action = random.choice(self.actions)
        self.prey_pos = self.move(self.prey_pos, prey_action)

        reward = -1
        done = False

        # Final capture check
        if self.predator_pos == self.prey_pos:
            reward = 10
            done = True

        return self.get_state(), reward, done

    def render(self):
        grid = [["." for _ in range(self.size)] for _ in range(self.size)]

        px, py = self.predator_pos
        qx, qy = self.prey_pos

        grid[px][py] = "P"  # Predator
        grid[qx][qy] = "R"  # pRey

        for row in grid:
            print(" ".join(row))
        print("\n")

    def get_state_space(self):
        return self.size * self.size * self.size * self.size

    def get_action_space(self):
        return len(self.actions)