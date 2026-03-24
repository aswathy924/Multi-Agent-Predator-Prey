from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.env.predator_prey import PredatorPreyEnv
from backend.agents.q_learning import QLearningAgent

import sys
import os
sys.path.append(os.path.dirname(__file__))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
env = PredatorPreyEnv()
agent = QLearningAgent()          # This will auto-load saved Q-table

episode_history = []

@app.get("/")
def home():
    return {"message": "RL Backend Running - Predator Prey Q-Learning"}

@app.get("/reset")
def reset():
    state = env.reset()
    return {"state": state}

@app.get("/step")
def step():
    state = env.get_state()

    action = agent.choose_action(state)
    next_state, reward, done = env.step(action)

    agent.update(state, action, reward, next_state)

    if done:
        agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)
        
        steps = env.steps
        episode_history.append(steps)

        # ✅ Save Q-table after every finished episode
        agent.save_q_table()

        env.episode += 1
        env.reset()

        return {
            "state": next_state,
            "done": True,
            "steps": steps,
            "episode": env.episode
        }

    return {
        "state": next_state,
        "done": False,
        "steps": env.steps,
        "episode": env.episode
    }

@app.post("/train")
def train(episodes: int = 500):
    """Train the agent for given number of episodes"""
    history = []
    for ep in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state

        agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)
        steps = env.steps
        history.append(steps)
        
        # Save after each training episode
        agent.save_q_table()

    return {
        "message": f"Training completed for {episodes} episodes",
        "episodes": list(range(1, len(history) + 1)),
        "steps": history
    }

@app.get("/metrics")
def metrics():
    return {
        "episodes": list(range(1, len(episode_history) + 1)),
        "steps": episode_history
    }