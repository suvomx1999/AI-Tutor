import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import torch
import os
import sys

# Fix import path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ai_tutor_rl'))

# Import our system components
from src.env import StudentEnv
from src.agent import DQNAgent

def run_agent(agent_type, num_episodes=50, max_steps=100):
    """
    Runs a simulation for a specific agent type.
    agent_type: 'random', 'rule_based', 'dqn'
    """
    env = StudentEnv(num_topics=5) # Smaller topic count for faster convergence in demo
    
    # Initialize DQN Agent if needed
    state_dim = 6
    action_dim = 5
    dqn_agent = None
    if agent_type == 'dqn':
        dqn_agent = DQNAgent(state_dim, action_dim)
        # Try to load pretrained model if exists, else it will train from scratch during this exp (or be random if epsilon high)
        if os.path.exists('models/dqn_tutor.pth'):
            try:
                dqn_agent.load('models/dqn_tutor.pth')
                dqn_agent.epsilon = 0.1 # Low epsilon for evaluation
            except:
                pass
    
    results = []

    print(f"Running Experiment: {agent_type.upper()}")
    
    for episode in tqdm(range(num_episodes)):
        state, _ = env.reset()
        total_reward = 0
        total_knowledge_gain = 0
        steps = 0
        
        # Track initial knowledge sum
        initial_knowledge = np.sum(env.student.knowledge)
        
        for step in range(max_steps):
            # 1. Select Action
            if agent_type == 'random':
                action = np.random.randint(0, 5)
            elif agent_type == 'rule_based':
                # Simple heuristic (similar to dashboard logic)
                score = state[2] * 10 # denormalize roughly
                diff = state[1]
                if score > 8.0: action = 1 # Harder
                elif score < 4.0: action = 0 # Easier
                elif score > 9.0: action = 4 # Next
                else: action = 3 # Practice
            elif agent_type == 'dqn':
                action = dqn_agent.get_action(state, eval_mode=True)
                
            # 2. Step
            next_state, reward, done, truncated, _ = env.step(action)
            
            total_reward += reward
            state = next_state
            steps += 1
            
            if done or truncated:
                break
                
        # End of Episode Metrics
        final_knowledge = np.sum(env.student.knowledge)
        knowledge_gain = final_knowledge - initial_knowledge
        # avg_score = env.student.get_average_score() # Removed as method doesn't exist
        
        results.append({
            'Agent': agent_type,
            'Episode': episode,
            'Total Reward': total_reward,
            'Knowledge Gain': knowledge_gain,
            # 'Final Avg Score': avg_score,
            'Steps Taken': steps
        })
        
    return pd.DataFrame(results)

def generate_paper_plots():
    # 1. Run Experiments
    # We compare 3 strategies to show "Comparative Analysis" (standard paper section)
    df_random = run_agent('random', num_episodes=30)
    df_rule = run_agent('rule_based', num_episodes=30)
    df_dqn = run_agent('dqn', num_episodes=30)
    
    # Combine data and reset index to avoid duplicate label errors
    df_all = pd.concat([df_random, df_rule, df_dqn]).reset_index(drop=True)
    
    # Set style
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Learning Efficiency (Knowledge Gain)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_all, x='Agent', y='Knowledge Gain', palette="Set2")
    plt.title("Comparative Learning Efficiency: Knowledge Gain per Episode")
    plt.ylabel("Total Knowledge Acquired (Sum across topics)")
    plt.savefig("paper_plot_knowledge_gain.png")
    print("Generated: paper_plot_knowledge_gain.png")
    
    # Plot 2: Reward Convergence
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_all, x='Episode', y='Total Reward', hue='Agent', style='Agent', markers=True, dashes=False)
    plt.title("Agent Performance: Average Reward Convergence")
    plt.ylabel("Cumulative Reward")
    plt.savefig("paper_plot_reward_convergence.png")
    print("Generated: paper_plot_reward_convergence.png")
    
    # Plot 3: Final Student Score Distribution
    # plt.figure(figsize=(10, 6))
    # sns.kdeplot(data=df_all, x='Final Avg Score', hue='Agent', fill=True, common_norm=False, alpha=0.4)
    # plt.title("Distribution of Final Student Scores")
    # plt.xlabel("Average Score (0-10)")
    # plt.savefig("paper_plot_score_dist.png")
    # print("Generated: paper_plot_score_dist.png")
    
    print("\n=== EXPERIMENT SUMMARY ===")
    print(df_all.groupby('Agent')[['Total Reward', 'Knowledge Gain']].mean())

if __name__ == "__main__":
    generate_paper_plots()
