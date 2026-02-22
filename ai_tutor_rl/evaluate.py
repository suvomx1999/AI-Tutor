import numpy as np
import matplotlib.pyplot as plt
import os
from src.env import StudentEnv
from src.agent import DQNAgent, DRQNAgent, LSTMTutorPolicy
from src.utils import smart_static_policy

def static_tutor_policy(state):
    # state: [topic, difficulty, score, time, failures, engagement]
    score = state[2]
    
    # Simple logic
    if score >= 8:
        return 1 # Harder
    elif score <= 4:
        return 0 # Easier
    else:
        return 3 # Practice
    
    # Note: This static policy doesn't handle "Next Topic" logic explicitly well, 
    # so we might add a condition: if high score and high difficulty, move next.
    # But for simplicity let's stick to this or slightly smarter.

#

def evaluate_agent(agent, env, episodes=10):
    total_rewards = []
    avg_knowledge_gains = []
    
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        initial_knowledge = np.mean(env.student.knowledge)
        
        while not (done or truncated):
            if agent is None:
                # Static Policy
                action = smart_static_policy(state, env.current_difficulty)
            else:
                action = agent.get_action(state, eval_mode=True)
                
            next_state, reward, done, truncated, _ = env.step(action)
            state = next_state
            episode_reward += reward
            
        total_rewards.append(episode_reward)
        avg_knowledge_gains.append(np.mean(env.student.knowledge) - initial_knowledge)
        
    return np.mean(total_rewards), np.mean(avg_knowledge_gains)

def visualize_episode(agent, filename):
    env = StudentEnv()
    state, _ = env.reset()
    done = False
    truncated = False
    
    topics = []
    difficulties = []
    scores = []
    actions = []
    steps = []
    
    print("\n--- Starting Visualization Episode ---")
    print(f"Initial State: Topic {env.current_topic}, Difficulty {env.current_difficulty}")
    
    step_count = 0
    while not (done or truncated):
        action = agent.get_action(state, eval_mode=True)
        
        # Capture pre-step info for transition logging
        prev_topic = env.current_topic
        
        next_state, reward, done, truncated, info = env.step(action)
        
        # Check for topic change
        if env.current_topic > prev_topic:
            print(f"Step {step_count}: ✅ TOPIC COMPLETED! Moving from Topic {prev_topic} to Topic {env.current_topic}")
            print(f"   -> Knowledge in Topic {prev_topic}: {env.student.knowledge[prev_topic]:.2f}")
        
        # Check for Final Topic Completion
        if env.current_topic == env.num_topics - 1 and env.student.knowledge[env.current_topic] > 0.9:
             if not getattr(env, 'final_topic_logged', False):
                 print(f"Step {step_count}: 🎓 FINAL TOPIC COMPLETED! Topic {env.current_topic} Mastered.")
                 print(f"   -> Knowledge in Topic {env.current_topic}: {env.student.knowledge[env.current_topic]:.2f}")
                 env.final_topic_logged = True

        steps.append(step_count)
        topics.append(env.current_topic)
        difficulties.append(env.current_difficulty)
        scores.append(env.last_score)
        actions.append(action)
        
        state = next_state
        step_count += 1
    
    print(f"Episode finished in {step_count} steps.")
    print(f"Final Knowledge State: {env.student.knowledge}")
    print("--------------------------------------\n")
        
    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Topic and Difficulty
    ax1.plot(steps, topics, label='Topic', marker='o', linestyle='--')
    ax1.plot(steps, difficulties, label='Difficulty', color='orange')
    ax1.set_ylabel('Topic / Difficulty')
    ax1.legend()
    ax1.set_title('Learning Path (Topic & Difficulty)')
    
    # Scores
    ax2.bar(steps, scores, alpha=0.6, color='green', label='Quiz Score')
    ax2.set_ylabel('Score')
    ax2.set_xlabel('Step')
    ax2.legend()
    ax2.set_title('Student Performance')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    env = StudentEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Load Agents
    dqn_agent = DQNAgent(state_dim, action_dim)
    dqn_model_path = 'models/dqn_tutor.pth'
    if os.path.exists(dqn_model_path):
        dqn_agent.load(dqn_model_path)
        print("DQN model loaded.")
    else:
        print("DQN model not found. Please train first.")
        exit()

    drqn_agent = DRQNAgent(state_dim, action_dim)
    drqn_model_path = 'models/drqn_tutor.pth'
    if os.path.exists(drqn_model_path):
        drqn_agent.load(drqn_model_path)
        print("DRQN model loaded.")
    else:
        print("DRQN model not found. You can still evaluate DQN and static baselines.")
    lstm_policy = LSTMTutorPolicy(state_dim, action_dim, seq_len=12, hidden_dim=128)
    lstm_model_path = 'models/lstm_tutor.pth'
    if os.path.exists(lstm_model_path):
        lstm_policy.load(lstm_model_path)
        print("LSTM Tutor model loaded.")
    else:
        print("LSTM Tutor model not found. Skipping.")
    
    # Compare
    print("Evaluating DQN Agent...")
    dqn_reward, dqn_gain = evaluate_agent(dqn_agent, env, episodes=20)
    if os.path.exists(drqn_model_path):
        print("Evaluating DRQN Agent...")
        drqn_reward, drqn_gain = evaluate_agent(drqn_agent, env, episodes=20)
    else:
        drqn_reward, drqn_gain = (np.nan, np.nan)
    
    print("Evaluating Static Tutor...")
    static_reward, static_gain = evaluate_agent(None, env, episodes=20)
    if os.path.exists(lstm_model_path):
        print("Evaluating LSTM Tutor...")
        # Wrap policy into an adapter to reuse evaluate_agent
        class Adapter:
            def get_action(self, state, eval_mode=True): return lstm_policy.get_action(state)
        lstm_reward, lstm_gain = evaluate_agent(Adapter(), env, episodes=20)
    else:
        lstm_reward, lstm_gain = (np.nan, np.nan)
    
    print(f"\nResults (Avg over 20 episodes):")
    print(f"DQN Agent - Reward: {dqn_reward:.2f}, Knowledge Gain: {dqn_gain:.4f}")
    print(f"DRQN Agent - Reward: {drqn_reward:.2f}, Knowledge Gain: {drqn_gain:.4f}")
    print(f"Static Tutor - Reward: {static_reward:.2f}, Knowledge Gain: {static_gain:.4f}")
    print(f"LSTM Tutor - Reward: {lstm_reward:.2f}, Knowledge Gain: {lstm_gain:.4f}")
    
    # Visualize one episode
    if not os.path.exists('plots'):
        os.makedirs('plots')
    visualize_episode(dqn_agent, 'plots/learning_path.png')
    print("Learning path visualization saved to plots/learning_path.png")
