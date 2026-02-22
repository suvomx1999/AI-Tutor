import numpy as np
import sys
import os
import torch
from tqdm import tqdm

# Add path to import src
sys.path.append(os.path.join(os.getcwd(), 'ai_tutor_rl'))

from src.env import StudentEnv
from src.student import StudentSimulator
from src.agent import DQNAgent, DRQNAgent, LSTMTutorPolicy
from src.utils import smart_static_policy

class RobustStudentSimulator(StudentSimulator):
    def __init__(self, profile, num_topics=5):
        super().__init__(num_topics)
        self.profile = profile
        
        # Default params (from original StudentSimulator)
        self.learning_rate = 0.25
        self.frustration_penalty = 0.05
        self.success_boost = 0.05
        self.noise_factor = 0.0 # Probability of random answer flip
        
        if profile == 'slow_learner':
            self.learning_rate = 0.10
            self.frustration_penalty = 0.10  # Gets frustrated easily
            self.knowledge = np.random.uniform(0.0, 0.2, size=num_topics) # Starts with low knowledge
            
        elif profile == 'fast_learner':
            self.learning_rate = 0.40
            self.frustration_penalty = 0.02
            self.knowledge = np.random.uniform(0.2, 0.5, size=num_topics)
            
        elif profile == 'disengaged':
            self.engagement = 0.4 # Starts bored
            self.success_boost = 0.02 # Hard to impress
            self.frustration_penalty = 0.15 # Easily quits
            self.noise_factor = 0.2 # 20% chance of answering randomly (guessing)
            
        elif profile == 'noisy':
            self.noise_factor = 0.3 # High randomness
            self.learning_rate = 0.25

    def attempt_question(self, topic_id, difficulty):
        # Override to inject noise for specific profiles
        is_correct, score, time_taken = super().attempt_question(topic_id, difficulty)
        
        # Inject noise for noisy/disengaged students (simulating guessing or misclicking)
        if np.random.random() < self.noise_factor:
            is_correct = not is_correct
            # Re-calculate score if flipped
            if is_correct:
                score = np.random.uniform(7.5, 10.0)
            else:
                score = np.random.uniform(0.0, 5.0)
            score = round(score, 1)
            
        return is_correct, score, time_taken

    def _update_state(self, is_correct, difficulty, topic_id):
        # Customized update logic
        old_k = self.knowledge[topic_id]
        if is_correct:
            gain = self.learning_rate * difficulty * self.engagement
            self.knowledge[topic_id] = min(1.0, self.knowledge[topic_id] + gain)
            self.engagement = min(1.0, self.engagement + self.success_boost)
        else:
            gain = self.learning_rate * 0.1
            self.knowledge[topic_id] = min(1.0, self.knowledge[topic_id] + gain)
            self.engagement = max(0.0, self.engagement - self.frustration_penalty)
            
        # print(f"Update: Correct={is_correct}, Diff={difficulty:.2f}, Gain={gain:.4f}, K_old={old_k:.2f}, K_new={self.knowledge[topic_id]:.2f}")
            
        self.fatigue += 0.001
        if self.fatigue > 0.8:
            self.engagement = max(0.0, self.engagement - 0.1)

# Custom Env to use RobustStudent
class RobustStudentEnv(StudentEnv):
    def __init__(self, profile='average', num_topics=5):
        self.profile = profile
        super().__init__(num_topics)
        
    def reset(self):
        self.student = RobustStudentSimulator(self.profile, self.num_topics)
        self.current_topic = 0
        self.current_difficulty = 0.5
        self.steps = 0
        self.max_steps = 300
        self.last_score = 0
        self.last_action = 0
        return self._get_obs(), {}

def evaluate_profile(agent, profile_name, episodes=50):
    env = RobustStudentEnv(profile=profile_name)
    
    gains = []
    engagements = []
    
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        init_k = np.mean(env.student.knowledge)
        ep_eng = []
        
        # Clear history for stateful agents
        if hasattr(agent, '_last_states'):
            agent._last_states.clear()
            
        while not done:
            ep_eng.append(env.student.engagement)
            if agent is None:
                # Rule Based
                action = smart_static_policy(state, env.current_difficulty)
            elif isinstance(agent, DQNAgent):
                # DQN
                action = agent.get_action(state, eval_mode=True)
            elif isinstance(agent, DRQNAgent):
                # DRQN
                action = agent.get_action(state, eval_mode=True)
                agent._last_states.append(state)
            elif isinstance(agent, LSTMTutorPolicy):
                # LSTM Tutor
                action = agent.get_action(state)
            else:
                 # Fallback
                 action = 3
                
            next_state, reward, done, truncated, _ = env.step(action)
            state = next_state
            
            if truncated: done = True
            
        final_k = np.mean(env.student.knowledge)
        gains.append(final_k - init_k)
        engagements.append(np.mean(ep_eng))
        
    return np.mean(gains), np.std(gains), np.mean(engagements)

def main():
    # Load Agents
    state_dim = 6
    action_dim = 5
    
    # DQN
    dqn_agent = DQNAgent(state_dim, action_dim)
    if os.path.exists('models/dqn_tutor.pth'):
        dqn_agent.load('models/dqn_tutor.pth')
        print("DQN Model loaded.")
    
    # DRQN
    drqn_agent = DRQNAgent(state_dim, action_dim, seq_len=12)
    if os.path.exists('models/drqn_tutor.pth'):
        drqn_agent.load('models/drqn_tutor.pth')
        print("DRQN Model loaded.")
        
    # LSTM Tutor
    lstm_agent = LSTMTutorPolicy(state_dim, action_dim, seq_len=12, hidden_dim=128)
    if os.path.exists('models/lstm_tutor.pth'):
        lstm_agent.load('models/lstm_tutor.pth')
        print("LSTM Tutor Model loaded.")

    profiles = ['fast_learner', 'slow_learner', 'disengaged', 'noisy']
    
    print(f"{'Profile':<15} | {'Agent':<10} | {'K. Gain':<15} | {'Eng.':<10}")
    print("-" * 60)
    
    results = {}
    
    for p in profiles:
        # Eval Rule Based
        k_rule, k_std_rule, eng_rule = evaluate_profile(None, p, episodes=20)
        print(f"{p:<15} | {'Rule':<10} | {k_rule:.2f} +/- {k_std_rule:.2f} | {eng_rule:.2f}")
        
        # Eval DQN
        k_dqn, k_std_dqn, eng_dqn = evaluate_profile(dqn_agent, p, episodes=20)
        print(f"{p:<15} | {'DQN':<10} | {k_dqn:.2f} +/- {k_std_dqn:.2f} | {eng_dqn:.2f}")

        # Eval DRQN
        k_drqn, k_std_drqn, eng_drqn = evaluate_profile(drqn_agent, p, episodes=20)
        print(f"{p:<15} | {'DRQN':<10} | {k_drqn:.2f} +/- {k_std_drqn:.2f} | {eng_drqn:.2f}")

        # Eval LSTM
        k_lstm, k_std_lstm, eng_lstm = evaluate_profile(lstm_agent, p, episodes=20)
        print(f"{p:<15} | {'LSTM':<10} | {k_lstm:.2f} +/- {k_std_lstm:.2f} | {eng_lstm:.2f}")
        print("-" * 60)
        
if __name__ == "__main__":
    main()
