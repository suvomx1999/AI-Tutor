import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curve(rewards, filename):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Episode Reward')
    # Moving average
    window_size = 50
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(rewards)), moving_avg, label='Moving Average (50)')
    
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def smart_static_policy(state, current_diff):
    score = state[2]
    if score > 8 and current_diff > 0.8:
        return 4 # Next topic
    elif score >= 7:
        return 1 # Harder
    elif score <= 4:
        return 0 # Easier
    else:
        return 3 # Practice
