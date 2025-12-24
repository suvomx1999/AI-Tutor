import numpy as np
import os
from src.env import StudentEnv
from src.agent import DQNAgent
from src.utils import plot_learning_curve

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

if __name__ == "__main__":
    train_dqn()
