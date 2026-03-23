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

episode_history = []
env = PredatorPreyEnv()
agent = QLearningAgent()

@app.get("/")
def home():
    return {"message": "RL Backend Running"}

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
        steps = env.steps
        episode_history.append(steps)   # ✅ store learning data

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

@app.get("/metrics")
def metrics():
    return {"episodes": list(range(1, len(episode_history)+1)),
            "steps": episode_history}