import asyncio
from fastapi import FastAPI, Body, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import time
import sys
import os
import sqlite3
import numpy as np

sys.path.append(os.path.dirname(__file__))

from env.continuous_env import ContinuousPreyEnv
from agents.dqn_agent import DQNAgent
from agents.d3qn_agent import D3QNAgent

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize SQLite Database
DB_FILE = "leaderboard.db"
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            score REAL NOT NULL,
            algorithm TEXT NOT NULL,
            predators INTEGER NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

env = ContinuousPreyEnv(num_predators=3, grid_size=1.0)
predator_agent = None
prey_agent = None

training = False
episode_rewards = []
training_task = None

manual_prey_action = 0
PLAY_MODE = False
CURRENT_ALGO = "d3qn"

def init_agents(num_predators, algorithm, difficulty):
    global env, predator_agent, prey_agent, CURRENT_ALGO
    
    env = ContinuousPreyEnv(num_predators=num_predators, grid_size=1.0)
    state_dim = 4 + (4 * num_predators)
    CURRENT_ALGO = algorithm
    
    if algorithm == "d3qn":
        predator_agent = D3QNAgent(state_dim=state_dim, action_dim=5, name="predator_d3qn", lr=0.001)
        prey_agent = D3QNAgent(state_dim=state_dim, action_dim=5, name="prey_d3qn", lr=0.001)
    else:
        predator_agent = DQNAgent(state_dim=state_dim, action_dim=5, name="predator_dqn", lr=0.001)
        prey_agent = DQNAgent(state_dim=state_dim, action_dim=5, name="prey_dqn", lr=0.001)

    if difficulty == "easy":
        predator_agent.epsilon = 0.8; predator_agent.epsilon_min = 0.5
        env.pred_max_speed = 0.03
    elif difficulty == "medium":
        predator_agent.epsilon = 0.2; predator_agent.epsilon_min = 0.1
        env.pred_max_speed = 0.05
    else:
        predator_agent.epsilon = 0.02; predator_agent.epsilon_min = 0.01
        env.pred_max_speed = 0.06

@app.on_event("startup")
def startup_event():
    init_agents(3, "d3qn", "medium")

@app.get("/")
def home():
    return {"message": "Deep MARL Sandbox Backend"}

@app.post("/init_game")
def init_game(payload: dict = Body(...)):
    num_preds = int(payload.get("num_predators", 3))
    algo = payload.get("algorithm", "d3qn")
    diff = payload.get("difficulty", "medium")
    init_agents(num_preds, algo, diff)
    return {"status": "ok", "num_predators": num_preds, "algorithm": algo}

@app.get("/state")
def get_state():
    return env.get_full_state()

@app.post("/action")
def set_action(payload: dict = Body(...)):
    global manual_prey_action, PLAY_MODE
    manual_prey_action = payload.get("action", 0)
    PLAY_MODE = payload.get("play_mode", False)
    return {"status": "ok"}

@app.post("/settings")
def set_settings(payload: dict = Body(...)):
    env.set_config(payload)
    if "pred_epsilon" in payload: predator_agent.epsilon = float(payload["pred_epsilon"])
    if "prey_epsilon" in payload: prey_agent.epsilon = float(payload["prey_epsilon"])
    return {"status": "ok"}

# --- LEADERBOARD ENDPOINTS ---
@app.get("/leaderboard")
def get_leaderboard():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT name, score, algorithm, predators 
        FROM scores 
        ORDER BY score DESC LIMIT 10
    ''')
    rows = cursor.fetchall()
    conn.close()
    leaderboard = [{"name": r[0], "score": r[1], "algorithm": r[2], "predators": r[3]} for r in rows]
    return {"leaderboard": leaderboard}

@app.post("/leaderboard/submit")
def submit_score(payload: dict = Body(...)):
    name = payload.get("name", "Unknown")
    score = payload.get("score", 0.0)
    algo = payload.get("algorithm", CURRENT_ALGO)
    preds = payload.get("predators", env.num_predators)
    
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO scores (name, score, algorithm, predators)
        VALUES (?, ?, ?, ?)
    ''', (name, score, algo, preds))
    conn.commit()
    conn.close()
    return {"status": "Score Submitted"}

# --- HEATMAP ENDPOINT FOR DEEP RL INSIGHTS ---
@app.get("/heatmap")
def get_heatmap():
    if not predator_agent:
        return {"heatmap": [], "grid_dim": 20}
        
    grid_dim = 20
    step_size = env.grid_size / grid_dim
    old_pos = env.pred_pos[0].copy()
    
    states_batch = []
    # Build 400 temporal states pretending agent 0 is all over the map
    # Loop over Y then X to form a 2D flat grid
    for y in range(grid_dim):
        for x in range(grid_dim):
            env.pred_pos[0] = np.array([x * step_size + step_size/2, y * step_size + step_size/2])
            states_batch.append(env.get_pred_state(0))
            
    env.pred_pos[0] = old_pos # Restore
    
    # Batched PyTorch Evaluation
    import torch
    with torch.no_grad():
        t_states = torch.FloatTensor(np.array(states_batch)).to(predator_agent.device)
        q_vals = predator_agent.policy_net(t_states)
        # Value of state V(s) is Max Q(s, a). Convert to native floats
        v_vals = q_vals.max(dim=1)[0].cpu().numpy().flatten().tolist()
        
    return {"heatmap": v_vals, "grid_dim": grid_dim}

async def training_loop():
    global training, manual_prey_action, PLAY_MODE
    episodes = 0
    while training:
        env.reset()
        done = False
        total_pred_r, total_prey_r = 0, 0
        
        while not done and training:
            pred_states = [env.get_pred_state(i) for i in range(env.num_predators)]
            prey_state = env.get_prey_state()
            
            pred_actions = [predator_agent.choose_action(s) for s in pred_states]
            
            if PLAY_MODE:
                prey_action = manual_prey_action
            else:
                prey_action = prey_agent.choose_action(prey_state)

            _, pred_r, prey_r, done, old_p, new_p, old_q, new_q = env.step(pred_actions, prey_action)

            total_pred_r += pred_r
            total_prey_r += prey_r

            for i in range(env.num_predators):
                predator_agent.store_transition(old_p[i], pred_actions[i], pred_r, new_p[i], done)
            if not PLAY_MODE:
                prey_agent.store_transition(old_q, prey_action, prey_r, new_q, done)

            predator_agent.train_step()
            if not PLAY_MODE: prey_agent.train_step()
            
            await asyncio.sleep(0.015) 
            
        if done:
            await asyncio.sleep(0.5)
            if PLAY_MODE: training = False
            else: await asyncio.sleep(0.5)

        episodes += 1
        episode_rewards.append({"episode": episodes, "predator_reward": total_pred_r, "prey_reward": total_prey_r, "survival_time": env.survival_time})
        if len(episode_rewards) > 100: episode_rewards.pop(0)

        predator_agent.decay_epsilon()
        if not PLAY_MODE: prey_agent.decay_epsilon()
        if episodes % 5 == 0:
            predator_agent.update_target_network()
            prey_agent.update_target_network()

@app.post("/train/start")
async def start_training(payload: dict = Body(default={})):
    global training, training_task, PLAY_MODE
    if training: return {"status": "Already active"}
    PLAY_MODE = payload.get("play_mode", False)
    training = True
    training_task = asyncio.create_task(training_loop())
    return {"status": "Started physics loop"}

@app.post("/train/stop")
def stop_training():
    global training
    training = False
    return {"status": "Stopped physics loop"}

@app.get("/metrics")
def get_metrics():
    p_loss = predator_agent.loss_history[-100:] if predator_agent.loss_history else []
    return {
        "training": training, "play_mode": PLAY_MODE, "algorithm": CURRENT_ALGO,
        "predator_epsilon": predator_agent.epsilon, "prey_epsilon": prey_agent.epsilon,
        "rewards": episode_rewards, "recent_loss": p_loss
    }