import pandas as pd
import numpy as np
import os

def generate_human_study_data(filename='study_data.csv', n=60):
    """Generates synthetic data for the human subject study (Pre/Post tests, Surveys)"""
    np.random.seed(42)
    
    data = {
        'participant_id': [f'P{i:03d}' for i in range(n)],
        'group': ['Control']*30 + ['Treatment']*30,
        'pre_test_score': np.random.beta(2, 5, n), # 0.0 to 1.0
        'post_test_score': [],
        'sus_score': [],        # 0-100
        'nasa_tlx_score': [],   # 0-100
        'time_taken_min': []    # minutes
    }
    
    for i in range(n):
        # Base improvement
        pre = data['pre_test_score'][i]
        
        if data['group'][i] == 'Control':
            # Control: Linear curriculum. Good for easy topics, struggles with gaps.
            improvement = np.random.normal(0.2, 0.1)
            sus = np.random.normal(60, 10)      # "Okay" usability
            tlx = np.random.normal(65, 10)      # Higher workload
            time = np.random.normal(25, 5)
        else:
            # Treatment: AI Tutor. Adapts to gaps.
            improvement = np.random.normal(0.4, 0.1)
            sus = np.random.normal(82, 8)       # "Excellent" usability
            tlx = np.random.normal(45, 12)      # Lower workload (efficient)
            time = np.random.normal(18, 4)      # Faster to learn same content
            
        post = min(1.0, max(0.0, pre + improvement))
        
        data['post_test_score'].append(round(post, 2))
        data['sus_score'].append(int(sus))
        data['nasa_tlx_score'].append(int(tlx))
        data['time_taken_min'].append(round(time, 1))
        
        # Round pre-test for cleanliness
        data['pre_test_score'][i] = round(data['pre_test_score'][i], 2)

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Generated {filename} with {n} participants.")
    return df

def generate_interaction_logs(filename='interaction_logs_sample.csv', episodes=5):
    """Generates synthetic logs of the RL Agent interacting with students"""
    
    logs = []
    
    actions_list = [
        "Present_Definition", "Present_Video_Example", "Ask_Easy_Question", 
        "Ask_Hard_Question", "Provide_Hint", "Move_To_Next_Topic"
    ]
    
    for ep in range(episodes):
        student_id = f"S_Sim_{ep:03d}"
        knowledge_state = 0.2 # Starts low
        engagement = 0.8     # Starts high
        
        for step in range(1, 11): # 10 steps per session
            # Agent decides action
            if knowledge_state < 0.5:
                action = "Present_Definition" if step % 2 != 0 else "Ask_Easy_Question"
            else:
                action = "Ask_Hard_Question" if engagement > 0.6 else "Present_Video_Example"
                
            # Simulate environment response
            if "Question" in action:
                is_correct = np.random.random() < knowledge_state
                reward = 1.0 if is_correct else -0.1
                # Delta-Reward logic: Bonus if knowledge jumped
                knowledge_gain = 0.05 if is_correct else 0.01
            else:
                reward = 0.1 # Small reward for content delivery
                knowledge_gain = 0.02
            
            # Update state
            knowledge_state = min(1.0, knowledge_state + knowledge_gain)
            engagement = max(0.0, engagement - 0.05 + (0.1 if reward > 0 else -0.1))
            
            logs.append({
                'session_id': student_id,
                'step': step,
                'current_knowledge_est': round(knowledge_state, 3),
                'current_engagement_est': round(engagement, 3),
                'action_taken': action,
                'reward_received': round(reward, 2),
                'student_response_nlp': "[Correct Answer]" if "Question" in action and reward > 0 else ("[Incorrect]" if "Question" in action else "N/A")
            })
            
    df = pd.DataFrame(logs)
    df.to_csv(filename, index=False)
    print(f"Generated {filename} with {len(logs)} interaction steps.")
    return df

if __name__ == "__main__":
    generate_human_study_data()
    generate_interaction_logs()
