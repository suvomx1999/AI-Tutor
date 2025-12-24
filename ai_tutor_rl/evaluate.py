import numpy as np
import matplotlib.pyplot as plt
import os
from src.env import StudentEnv
from src.agent import DQNAgent

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

def smart_static_policy(state, current_diff):
    score = state[2]
    # If mastered (high score on high difficulty), move next
    if score > 8 and current_diff > 0.8:
        return 4 # Next topic
    elif score >= 7:
        return 1 # Harder
    elif score <= 4:
        return 0 # Easier
    else:
        return 3 # Practice

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
    
    # Load Agent
    agent = DQNAgent(state_dim, action_dim)
    model_path = 'models/dqn_tutor.pth'
    if os.path.exists(model_path):
        agent.load(model_path)
        print("Model loaded.")
    else:
        print("Model not found. Please train first.")
        exit()
        
    # Compare
    print("Evaluating RL Agent...")
    rl_reward, rl_gain = evaluate_agent(agent, env, episodes=20)
    
    print("Evaluating Static Tutor...")
    static_reward, static_gain = evaluate_agent(None, env, episodes=20)
    
    print(f"\nResults (Avg over 20 episodes):")
    print(f"RL Agent - Reward: {rl_reward:.2f}, Knowledge Gain: {rl_gain:.4f}")
    print(f"Static Tutor - Reward: {static_reward:.2f}, Knowledge Gain: {static_gain:.4f}")
    
    # Visualize one episode
    if not os.path.exists('plots'):
        os.makedirs('plots')
    visualize_episode(agent, 'plots/learning_path.png')
    print("Learning path visualization saved to plots/learning_path.png")
