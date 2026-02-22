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
        # Action Masking Logic
        # State: [topic (norm), diff, score (norm), time, fail, eng]
        # Actions: 0: Easier, 1: Harder, 2: Revision, 3: Practice, 4: Next Topic
        
        valid_actions = [0, 1, 2, 3, 4]
        
        # Unpack state for readability (assuming normalized inputs)
        # Note: In real training, state is a numpy array. 
        # We need raw values for logic, but here we estimate from normalized state or passed raw state.
        # For simplicity, we apply heuristic masking on the q_values directly.
        
        current_diff = state[1]
        last_score = state[2] * 10.0 # Denormalize
        
        # Mask 1: Don't increase difficulty if already maxed (or close)
        if current_diff >= 1.0:
            if 1 in valid_actions: valid_actions.remove(1)
            
        # Mask 2: Don't decrease difficulty if already min (or close)
        if current_diff <= 0.1:
            if 0 in valid_actions: valid_actions.remove(0)
            
        # Mask 3: Don't move to Next Topic if score is low
        if last_score < 5.0:
            if 4 in valid_actions: valid_actions.remove(4)
            
        if not valid_actions: # Fallback if all masked (shouldn't happen)
            valid_actions = [3] # Default to Practice

        if not eval_mode and np.random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            
        # Set invalid actions to -inf so argmax won't pick them
        mask = torch.full_like(q_values, float('-inf'))
        mask[0, valid_actions] = 0 # Unmask valid ones
        
        masked_q_values = q_values + mask
        return masked_q_values.argmax().item()

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

# --------------------------
# DRQN Agent (LSTM Q-Network)
# --------------------------

class DRQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(DRQN, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, hidden=None):
        # x: (batch, seq_len, input_dim)
        out, hidden = self.lstm(x, hidden)
        # Take last timestep
        last = out[:, -1, :]
        q = self.fc(last)
        return q, hidden

class DRQNAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.99, min_epsilon=0.05, batch_size=64, memory_size=20000, seq_len=12):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.seq_len = seq_len

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DRQN(state_dim, action_dim).to(self.device)
        self.target_net = DRQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        for p in self.target_net.parameters():
            p.requires_grad = False

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.memory = deque(maxlen=memory_size)
        self.steps = 0
        self.update_target_every = 1500

        self._last_states = deque(maxlen=self.seq_len)
        self._episodes = deque(maxlen=1000)
        self._current_episode = []

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        self._last_states.append(state)
        self._current_episode.append((state, action, reward, next_state, done))
        if done:
            self._episodes.append(self._current_episode)
            self._current_episode = []

    def _valid_actions(self, state):
        valid = [0, 1, 2, 3, 4]
        current_diff = state[1]
        last_score = state[2] * 10.0
        if current_diff >= 1.0 and 1 in valid:
            valid.remove(1)
        if current_diff <= 0.1 and 0 in valid:
            valid.remove(0)
        if last_score < 5.0 and 4 in valid:
            valid.remove(4)
        if not valid:
            valid = [3]
        return valid

    def get_action(self, state, eval_mode=False):
        valid_actions = self._valid_actions(state)
        if not eval_mode and np.random.random() < self.epsilon:
            return random.choice(valid_actions)

        # Build sequence input from last states plus current
        seq = list(self._last_states) + [state]
        while len(seq) < self.seq_len:
            seq.insert(0, seq[0])
        seq = np.array(seq[-self.seq_len:])
        seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(self.device)  # (1, seq_len, state_dim)

        with torch.no_grad():
            q_values, _ = self.policy_net(seq_tensor)
        mask = torch.full_like(q_values, float('-inf'))
        mask[0, valid_actions] = 0
        return (q_values + mask).argmax().item()

    def replay(self):
        if len(self._episodes) < 1:
            return
        episodes = random.sample(list(self._episodes), min(len(self._episodes), self.batch_size))
        batch_seqs = []
        batch_actions = []
        batch_rewards = []
        batch_next_seqs = []
        batch_dones = []

        for ep in episodes:
            if len(ep) < self.seq_len:
                continue
            start = random.randint(0, len(ep) - self.seq_len)
            seq_trans = ep[start:start + self.seq_len]
            states = [t[0] for t in seq_trans]
            actions = [t[1] for t in seq_trans]
            rewards = [t[2] for t in seq_trans]
            next_states = [t[3] for t in seq_trans]
            dones = [t[4] for t in seq_trans]
            batch_seqs.append(states)
            batch_actions.append(actions[-1])  # last action
            batch_rewards.append(rewards[-1])
            batch_next_seqs.append(next_states)
            batch_dones.append(dones[-1])

        seq_tensor = torch.FloatTensor(np.array(batch_seqs)).to(self.device)  # (B, seq, dim)
        actions_t = torch.LongTensor(batch_actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(batch_rewards).unsqueeze(1).to(self.device)
        next_seq_tensor = torch.FloatTensor(np.array(batch_next_seqs)).to(self.device)
        dones_t = torch.FloatTensor(batch_dones).unsqueeze(1).to(self.device)

        current_q, _ = self.policy_net(seq_tensor)
        current_q = current_q.gather(1, actions_t)

        with torch.no_grad():
            next_q_policy, _ = self.policy_net(next_seq_tensor)
            next_q_target, _ = self.target_net(next_seq_tensor)
            masked_next_q_policy = []
            for i in range(next_seq_tensor.size(0)):
                valid = self._valid_actions(batch_next_seqs[i][-1])
                mask_vec = torch.full((self.action_dim,), float('-inf'), device=self.device)
                mask_vec[valid] = 0
                masked_next_q_policy.append(next_q_policy[i] + mask_vec)
            masked_next_q_policy = torch.stack(masked_next_q_policy, dim=0)
            next_actions = masked_next_q_policy.argmax(1).unsqueeze(1)
            max_next_q = next_q_target.gather(1, next_actions)
            target_q = rewards_t + (self.gamma * max_next_q * (1 - dones_t))

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def end_episode(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

# --------------------------
# Supervised LSTM Tutor
# --------------------------
class LSTMTutor(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=64):
        super(LSTMTutor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        logits = self.head(last)
        return logits

class LSTMTutorPolicy:
    def __init__(self, state_dim, action_dim, seq_len=8, hidden_dim=64):
        self.seq_len = seq_len
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LSTMTutor(state_dim, action_dim, hidden_dim).to(self.device)
        self._last_states = deque(maxlen=seq_len)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def _valid_actions(self, state):
        valid = [0,1,2,3,4]
        current_diff = state[1]
        last_score = state[2] * 10.0
        if current_diff >= 1.0 and 1 in valid: valid.remove(1)
        if current_diff <= 0.1 and 0 in valid: valid.remove(0)
        if last_score < 5.0 and 4 in valid: valid.remove(4)
        if not valid: valid=[3]
        return valid

    def get_action(self, state, eval_mode=False):
        self._last_states.append(state)
        seq = list(self._last_states)
        if len(seq) == 0:
            return 3
        while len(seq) < self.seq_len:
            seq.insert(0, seq[0])
        seq = np.array(seq[-self.seq_len:])
        tensor = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        valid = self._valid_actions(state)
        # zero out invalid
        masked = np.full_like(probs, -np.inf)
        masked[valid] = probs[valid]
        return int(np.argmax(masked))
