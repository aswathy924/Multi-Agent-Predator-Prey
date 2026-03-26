import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

MODEL_DIR = "backend/agents/models"

class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingQNetwork, self).__init__()
        
        # Common Feature Extractor
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Value Stream V(s) - Predicts how good the state is overall
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Advantage Stream A(s, a) - Predicts how much better an action is relative to others
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, state):
        features = self.feature_layer(state)
        
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        # This forces the advantage stream to zero out on average, solving the identifiability problem
        q_vals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_vals

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones)
        )

    def __len__(self):
        return len(self.buffer)

class D3QNAgent:
    def __init__(self, state_dim, action_dim=5, name="d3qn_agent", lr=5e-4, gamma=0.99):
        self.name = name
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        
        self.batch_size = 64
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = DuelingQNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DuelingQNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer()
        self.loss_history = []
        
        self.load_model()

    def choose_action(self, state, evaluate=False):
        if not evaluate and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t)
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return 0.0
            
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.unsqueeze(1).to(self.device)
        rewards = rewards.unsqueeze(1).to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.unsqueeze(1).to(self.device)

        # 1. Action Selection: Q_local(s') -> argmax a'
        q_values_next = self.policy_net(next_states)
        best_actions_next = q_values_next.argmax(dim=1, keepdim=True)
        
        # 2. Action Evaluation: Q_target(s', best_a')
        # This is the DOUBLE DQN component! Stops overestimation bias.
        with torch.no_grad():
            q_target_next = self.target_net(next_states).gather(1, best_actions_next)
            target_q = rewards + (1 - dones) * self.gamma * q_target_next

        # 3. Current Q values
        q_values = self.policy_net(states).gather(1, actions)

        # 4. Compute Loss
        loss = nn.SmoothL1Loss()(q_values, target_q) # Huber loss for stability
        
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        loss_val = loss.item()
        self.loss_history.append(loss_val)
        return loss_val

    def update_target_network(self, tau=0.005):
        # Soft update for stability
        for target_param, local_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_model(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        path = os.path.join(MODEL_DIR, f"{self.name}.pth")
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self):
        path = os.path.join(MODEL_DIR, f"{self.name}.pth")
        if os.path.exists(path):
            self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print(f"✅ Loaded {self.name} D3QN model.")
        else:
            print(f"ℹ️ No D3QN model found for {self.name}.")
