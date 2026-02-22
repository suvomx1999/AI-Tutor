import numpy as np
import torch
import os
import sys
sys.path.append('ai_tutor_rl')
from src.env import StudentEnv
from src.agent import DQNAgent, DRQNAgent, LSTMTutorPolicy

def evaluate_agent(agent, n_episodes=20):
    env = StudentEnv()
    env.max_steps = 1000
    gains = []
    engagements = []
    times = []
    
    for _ in range(n_episodes):
        state, _ = env.reset()
        
        # Reset agent memory
        if hasattr(agent, '_last_states'):
            agent._last_states.clear()
            
        initial_knowledge = np.mean(env.student.knowledge)
        total_eng = 0
        steps = 0
        done = False
        
        while not done:
            if isinstance(agent, LSTMTutorPolicy):
                action = agent.get_action(state)
            else:
                action = agent.get_action(state, eval_mode=True)
                # Manually update history for DRQN in eval mode
                if isinstance(agent, DRQNAgent):
                    agent._last_states.append(state)
                
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state
            total_eng += info['engagement']
            steps += 1
            
        final_knowledge = np.mean(env.student.knowledge)
        gains.append(final_knowledge - initial_knowledge)
        final_ks.append(final_knowledge)
        engagements.append(total_eng / steps)
        times.append(steps) # Proxy for time
        
    return np.mean(gains), np.std(gains), np.mean(engagements), np.std(engagements), np.mean(times), np.mean(final_ks)

def main():
    state_dim = 6
    action_dim = 5
    
    # DQN
    dqn_agent = DQNAgent(state_dim, action_dim)
    if os.path.exists('models/dqn_tutor.pth'):
        dqn_agent.load('models/dqn_tutor.pth')
    
    # DRQN
    drqn_agent = DRQNAgent(state_dim, action_dim, seq_len=12)
    if os.path.exists('models/drqn_tutor.pth'):
        drqn_agent.load('models/drqn_tutor.pth')
        
    # LSTM Tutor
    lstm_agent = LSTMTutorPolicy(state_dim, action_dim, seq_len=12, hidden_dim=128)
    if os.path.exists('models/lstm_tutor.pth'):
        lstm_agent.load('models/lstm_tutor.pth')

    print(f"{'Agent':<10} | {'K. Gain':<15} | {'Eng.':<10} | {'Time':<10} | {'Final K':<10}")
    
    # DQN
    k_dqn, k_std_dqn, eng_dqn, eng_std_dqn, time_dqn, final_dqn = evaluate_agent(dqn_agent)
    print(f"{'DQN':<10} | {k_dqn:.2f} +/- {k_std_dqn:.2f} | {eng_dqn:.2f} | {time_dqn:.1f} | {final_dqn:.2f}")

    # DRQN
    k_drqn, k_std_drqn, eng_drqn, eng_std_drqn, time_drqn, final_drqn = evaluate_agent(drqn_agent)
    print(f"{'DRQN':<10} | {k_drqn:.2f} +/- {k_std_drqn:.2f} | {eng_drqn:.2f} | {time_drqn:.1f} | {final_drqn:.2f}")

    # LSTM
    k_lstm, k_std_lstm, eng_lstm, eng_std_lstm, time_lstm, final_lstm = evaluate_agent(lstm_agent)
    print(f"{'LSTM':<10} | {k_lstm:.2f} +/- {k_std_lstm:.2f} | {eng_lstm:.2f} | {time_lstm:.1f} | {final_lstm:.2f}")

if __name__ == "__main__":
    main()
