import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# --------------------------
# Q-Learning Agent (Tabular)
# --------------------------
class QLearningAgent:
    def __init__(self, action_space_size, state_bins, learning_rate=0.1, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        self.action_space_size = action_space_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        # Q-Table initialization
        # We need to map continuous state to discrete bins
        self.bins = state_bins
        # Calculate table size: product of bins for each dimension
        self.state_dims = [len(b) + 1 for b in self.bins]
        self.q_table = np.zeros(self.state_dims + [action_space_size])

    def _discretize_state(self, state):
        # state: [topic, difficulty, score, time, failures, engagement]
        discrete_state = []
        for i, val in enumerate(state):
            # np.digitize returns index of bin
            idx = np.digitize(val, self.bins[i])
            # clip to ensure valid index (though digitize usually handles it)
            idx = min(idx, self.state_dims[i] - 1)
            discrete_state.append(idx)
        return tuple(discrete_state)

    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_space_size)
        
        discrete_state = self._discretize_state(state)
        return np.argmax(self.q_table[discrete_state])

    def update(self, state, action, reward, next_state, done):
        discrete_state = self._discretize_state(state)
        discrete_next_state = self._discretize_state(next_state)
        
        current_q = self.q_table[discrete_state][action]
        max_next_q = np.max(self.q_table[discrete_next_state])
        
        target = reward + (self.gamma * max_next_q * (1 - done))
        error = target - current_q
        
        self.q_table[discrete_state][action] += self.lr * error
        
        if done:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

# --------------------------
# DQN Agent (Deep Q-Network)
# --------------------------

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        
    def forward(self, x):
        return self.fc(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01, batch_size=64, memory_size=10000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
        self.memory = deque(maxlen=memory_size)
        self.steps = 0
        self.update_target_every = 1000

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state, eval_mode=False):
        if not eval_mode and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return q_values.argmax().item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Q(s, a)
        current_q = self.policy_net(states).gather(1, actions)
        
        # Max Q(s', a') from target net
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (self.gamma * next_q * (1 - dones))
            
        loss = self.loss_fn(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.steps += 1
        if self.steps % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
            
    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)
        
    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())
