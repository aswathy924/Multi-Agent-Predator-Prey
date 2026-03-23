from environment.predator_prey import PredatorPreyEnv
from agents.q_learning_agent import QLearningAgent

def train_predator(episodes=1000):
    env = PredatorPreyEnv(size=5)

    state_size = env.get_state_space()
    action_size = env.get_action_space()

    agent = QLearningAgent(state_size, action_size)

    rewards = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            state_idx = agent.state_to_index(state, env.size)

            action = agent.choose_action(state_idx)
            action_name = env.actions[action]

            next_state, reward, done = env.step(action_name)

            next_state_idx = agent.state_to_index(next_state, env.size)

            agent.update(state_idx, action, reward, next_state_idx)

            state = next_state
            total_reward += reward

        agent.epsilon *= 0.995
        rewards.append(total_reward)

        if episode % 100 == 0:
            print(f"Episode {episode}, Reward: {total_reward}")

    return agent, rewards