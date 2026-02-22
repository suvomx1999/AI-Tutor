import numpy as np
import os
from src.env import StudentEnv
from src.agent import DQNAgent, DRQNAgent, LSTMTutor, LSTMTutorPolicy
from src.utils import plot_learning_curve, smart_static_policy
import torch
import torch.nn as nn
import torch.optim as optim

def train_dqn(episodes=500):
    env = StudentEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DQNAgent(state_dim, action_dim)
    
    rewards_history = []
    
    print(f"Starting training for {episodes} episodes...")
    
    for e in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            action = agent.get_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            
            state = next_state
            total_reward += reward
            
        rewards_history.append(total_reward)
        
        if (e + 1) % 10 == 0:
            print(f"Episode {e+1}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
            
    # Save model
    if not os.path.exists('models'):
        os.makedirs('models')
    agent.save('models/dqn_tutor.pth')
    
    # Plot results
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plot_learning_curve(rewards_history, 'plots/dqn_training.png')
    
    print("Training finished.")
    return agent

def collect_demonstrations(num_episodes=300, seq_len=8):
    env = StudentEnv()
    dataset = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        window = []
        ep_reward = 0.0
        ep_samples = []
        while not (done or truncated):
            action = smart_static_policy(state, env.current_difficulty)
            next_state, reward, done, truncated, _ = env.step(action)
            ep_reward += reward
            window.append(state)
            if len(window) > seq_len:
                window.pop(0)
            if len(window) == seq_len:
                ep_samples.append((np.array(window), action))
            state = next_state
        w = 0.5
        for x, a in ep_samples:
            dataset.append((x, a, w))
    return dataset

def collect_demonstrations_dqn(model_path='models/dqn_tutor.pth', num_episodes=300, seq_len=12):
    if not os.path.exists(model_path):
        print("DQN model not found; skipping DQN demonstrations.")
        return []
    env = StudentEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim)
    agent.load(model_path)
    dataset = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        window = []
        ep_reward = 0.0
        ep_samples = []
        while not (done or truncated):
            action = agent.get_action(state, eval_mode=True)
            next_state, reward, done, truncated, _ = env.step(action)
            ep_reward += reward
            window.append(state)
            if len(window) > seq_len:
                window.pop(0)
            if len(window) == seq_len:
                ep_samples.append((np.array(window), action))
            state = next_state
        w = 1.0
        for x, a in ep_samples:
            dataset.append((x, a, w))
    return dataset

def train_lstm_tutor(seq_len=8, hidden_dim=64, epochs=10, batch_size=64):
    dataset_static = collect_demonstrations(num_episodes=200, seq_len=seq_len)
    dataset_dqn = collect_demonstrations_dqn(num_episodes=800, seq_len=seq_len)
    dataset = dataset_dqn + dataset_static
    if len(dataset) == 0:
        print("No demonstrations collected.")
        return None
    state_dim = dataset[0][0].shape[-1]
    action_dim = 5
    model = LSTMTutor(state_dim, action_dim, hidden_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    # Prepare tensors
    def batch_gen(data, bs):
        idx = np.random.permutation(len(data))
        for i in range(0, len(data), bs):
            j = idx[i:i+bs]
            X = np.stack([data[k][0] for k in j])
            y = np.array([data[k][1] for k in j])
            w = np.array([data[k][2] for k in j], dtype=np.float32)
            yield torch.FloatTensor(X).to(device), torch.LongTensor(y).to(device), torch.FloatTensor(w).to(device)
    for ep in range(epochs):
        total_loss = 0.0
        steps = 0
        for X, y, w in batch_gen(dataset, batch_size):
            optimizer.zero_grad()
            logits = model(X)
            l = loss_fn(logits, y)
            loss = torch.mean(l * w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            steps += 1
        print(f"[LSTM Tutor] Epoch {ep+1}/{epochs}, Loss: {total_loss/max(1,steps):.4f}")
    # Save
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(model.state_dict(), 'models/lstm_tutor.pth')
    print("LSTM Tutor training finished.")
    return model
def train_drqn(episodes=500):
    env = StudentEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DRQNAgent(state_dim, action_dim)
    rewards_history = []
    print(f"Starting DRQN training for {episodes} episodes...")
    for e in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        truncated = False
        while not (done or truncated):
            action = agent.get_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            state = next_state
            total_reward += reward
        rewards_history.append(total_reward)
        if (e + 1) % 10 == 0:
            print(f"[DRQN] Episode {e+1}/{episodes}, Reward: {total_reward:.2f}, Eps: {agent.epsilon:.2f}")
        agent.end_episode()
    if not os.path.exists('models'):
        os.makedirs('models')
    agent.save('models/drqn_tutor.pth')
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plot_learning_curve(rewards_history, 'plots/drqn_training.png')
    print("DRQN training finished.")
    return agent
if __name__ == "__main__":
    train_dqn(episodes=500)
    train_drqn(episodes=500)
    train_lstm_tutor(epochs=10)
