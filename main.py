# from training.train_agent import train
# from environment.gridworld import GridWorld
from training.train_predator import train_predator
from environment.predator_prey import PredatorPreyEnv

import matplotlib.pyplot as plt
import time

# agent, rewards = train(episodes=1000)

# plt.plot(rewards)
# plt.title("Training Rewards")
# plt.xlabel("Episode")
# plt.ylabel("Reward")
# plt.savefig("output.png")
# plt.show()

# # Turn off exploration
# agent.epsilon = 0

# env = GridWorld(size=5)
# state = env.reset()

# done = False

# print("=== Learned Path ===")

# while not done:
#     env.render()

#     state_idx = agent.state_to_index(state, env.size)
#     action = agent.choose_action(state_idx)

#     action_name = env.actions[action]
#     state, _, done = env.step(action_name)

# # show final state
# env.render()


agent, rewards = train_predator(episodes=1500)

plt.plot(rewards)
plt.title("Predator Training Rewards")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.savefig("Predator output")
plt.show()

agent.epsilon = 0  # no exploration

env = PredatorPreyEnv(size=5)
state = env.reset()

done = False

print("\n\n=== Predator Chasing Prey ===")

while not done:
    env.render()
    time.sleep(0.4)

    state_idx = agent.state_to_index(state, env.size)
    action = agent.choose_action(state_idx)

    action_name = env.actions[action]
    state, _, done = env.step(action_name)

env.render()


