from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
import os
from src.agent import DQNAgent

app = FastAPI(title="AI Tutor RL API", description="Real-time RL-based tutoring recommendations for integration with LMS.")

# Load Model
NUM_TOPICS = 10
STATE_DIM = 6
ACTION_DIM = 5
HIDDEN_DIM = 128

agent = DQNAgent(STATE_DIM, ACTION_DIM, HIDDEN_DIM)
MODEL_PATH = "models/dqn_tutor.pth"

if os.path.exists(MODEL_PATH):
    agent.load(MODEL_PATH)
    print(f"✅ Model loaded from {MODEL_PATH}")
else:
    print("⚠️ Warning: Model not found. Recommendations will be random.")

class StudentState(BaseModel):
    current_topic: int
    current_difficulty: float
    last_score: float # 0-10
    last_time: float # seconds
    consecutive_failures: int
    engagement: float # 0.0-1.0

class Recommendation(BaseModel):
    action_id: int
    action_name: str
    description: str

ACTION_MAP = {
    0: ("Easier Question", "Reduce difficulty to build confidence."),
    1: ("Harder Question", "Increase difficulty to challenge student."),
    2: ("Revision", "Review material to consolidate knowledge."),
    3: ("Practice", "Maintain current difficulty."),
    4: ("Next Topic", "Advance to the next topic.")
}

@app.get("/recommend")
def recommend_info():
    return {"message": "Please use POST method with StudentState body to get recommendations."}

@app.post("/recommend", response_model=Recommendation)
def get_recommendation(state: StudentState):
    """
    Get the next best tutoring action based on student state.
    """
    # Convert state to numpy array matching observation space (NORMALIZED)
    obs = np.array([
        state.current_topic / float(NUM_TOPICS),
        state.current_difficulty,
        state.last_score / 10.0,
        state.last_time / 60.0,
        state.consecutive_failures / 5.0,
        state.engagement
    ], dtype=np.float32)
    
    # Hybrid Logic: Rule-Based Overrides + RL Agent
    # If the decision is obvious, don't trust the potentially unstable RL agent.
    
    action = None
    
    # Rule 1: Mastered Topic -> Move Next
    if state.last_score >= 9.0 and state.current_difficulty >= 0.8:
         action = 4 # Next Topic
         
    # Rule 2: Cruising -> Increase Difficulty
    elif state.last_score >= 8.0 and state.current_difficulty < 0.9:
        action = 1 # Harder Question
        
    # Rule 3: Struggling -> Decrease Difficulty (RL is usually good at this too)
    elif (state.consecutive_failures >= 2 or state.last_score < 4.0) and state.current_difficulty > 0.15:
        action = 0 # Easier Question
        
    # Rule 4: Absolute Bottom -> Remedial Action
    # If they are failing the EASIEST difficulty (0.1), giving them "Easier Question" (Action 0) won't help.
    # We must switch to "Revision" (Action 2) to help them learn.
    elif state.last_score < 5.0 and state.current_difficulty <= 0.15:
        action = 2 # Revision / Study Material

    # Fallback to RL Agent if no obvious rule applies
    if action is None:
        action = agent.get_action(obs, eval_mode=True)
    
    # Map to human-readable format
    name, desc = ACTION_MAP.get(action, ("Unknown", "Unknown Action"))
    
    return Recommendation(action_id=action, action_name=name, description=desc)

@app.get("/")
def health_check():
    return {"status": "active", "service": "AI Tutor RL Agent"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
