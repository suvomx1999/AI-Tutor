import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Ensure src can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.env import StudentEnv
from src.agent import DQNAgent, DRQNAgent
from cross_dataset_loader import get_dataset_params

def evaluate_on_dataset(agent, dataset_name, episodes=30):
    params = get_dataset_params(dataset_name)
    env = StudentEnv(student_config=params)
    
    knowledge_gains = []
    
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        initial_knowledge = np.mean(env.student.knowledge)
        
        while not (done or truncated):
            # Handle RandomAgent dummy
            if hasattr(agent, 'get_action'):
                action = agent.get_action(state, eval_mode=True)
            else:
                action = np.random.randint(0, 5)
                
            next_state, reward, done, truncated, _ = env.step(action)
            state = next_state
            
        final_knowledge = np.mean(env.student.knowledge)
        knowledge_gains.append(final_knowledge - initial_knowledge)
        
    return np.mean(knowledge_gains), np.std(knowledge_gains)

def run_cross_dataset_eval():
    datasets = ['oulad', 'assistments', 'ednet', 'junyi']
    
    # Load DRQN Agent (Best performing)
    env_dummy = StudentEnv()
    state_dim = env_dummy.observation_space.shape[0]
    action_dim = env_dummy.action_space.n
    
    agent = DRQNAgent(state_dim, action_dim)
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'drqn_tutor.pth')
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Using Random Agent for testing.")
        # Create a dummy agent if model missing (just for testing flow)
        class RandomAgent:
            def get_action(self, state, eval_mode=True):
                return np.random.randint(0, action_dim)
        agent = RandomAgent()
    else:
        agent.load(model_path)
        print("DRQN Model loaded successfully.")
    
    results = {}
    
    print("\nStarting Cross-Dataset Evaluation...")
    print("-" * 60)
    
    for ds in datasets:
        mean_gain, std_gain = evaluate_on_dataset(agent, ds)
        results[ds] = (mean_gain, std_gain)
        print(f"Dataset: {ds:<15} | Knowledge Gain: {mean_gain:.3f} (+/- {std_gain:.3f})")
        
    # Plotting
    names = [d.upper() for d in datasets]
    means = [results[d][0] for d in datasets]
    stds = [results[d][1] for d in datasets]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, means, yerr=stds, capsize=10, color=['#4c72b0', '#55a868', '#c44e52', '#8172b3'], alpha=0.8)
    
    plt.ylabel('Knowledge Gain (0-1)')
    plt.title('Cross-Dataset Generalization (Simulated)')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{height:.2f}',
                 ha='center', va='bottom')
                 
    output_path = 'paper_plot_cross_dataset.png'
    plt.savefig(output_path, dpi=300)
    print(f"\nSaved plot to {output_path}")

if __name__ == "__main__":
    run_cross_dataset_eval()
