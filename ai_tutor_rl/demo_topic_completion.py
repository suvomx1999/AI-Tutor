from src.env import StudentEnv
import numpy as np

def demo_topic_completion():
    env = StudentEnv()
    env.reset()
    
    print("\n--- DEMONSTRATION: Topic Completion Sequence ---")
    print(f"Start: Topic {env.current_topic}, Difficulty {env.current_difficulty}")
    
    # Simulate a few steps in Topic 0
    print("\n[Step 1-3] Student answers questions in Topic 0...")
    for _ in range(3):
        env.step(1) # Increase difficulty / normal learning
        
    print(f"Current State: Topic {env.current_topic}, Knowledge: {env.student.knowledge[0]:.2f}")
    
    # FORCE Action 4: Next Topic
    print("\n>>> AGENT ACTION: Move to Next Topic (Action 4) <<<")
    
    prev_topic = env.current_topic
    obs, reward, done, truncated, info = env.step(4)
    
    print(f"\n✅ OUTPUT TRIGGERED:")
    print(f"1. Topic Index: {prev_topic} -> {env.current_topic}")
    print(f"2. Difficulty Reset: {env.current_difficulty} (Default is 0.5)")
    print(f"3. Reward Received: {reward:.2f}")
    print(f"4. New Topic Knowledge: {env.student.knowledge[env.current_topic]:.2f}")
    
    print("\nSystem is now ready for Topic 1 content.")

if __name__ == "__main__":
    demo_topic_completion()
