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

@app.post("/recommend", response_model=Recommendation)
def get_recommendation(state: StudentState):
    """
    Get the next best tutoring action based on student state.
    """
    # Convert state to numpy array matching observation space
    obs = np.array([
        state.current_topic,
        state.current_difficulty,
        state.last_score,
        state.last_time,
        state.consecutive_failures,
        state.engagement
    ], dtype=np.float32)
    
    # Get action from agent (deterministic for deployment)
    action = agent.act(obs, epsilon=0.0)
    
    # Map to human-readable format
    name, desc = ACTION_MAP.get(action, ("Unknown", "Unknown Action"))
    
    return Recommendation(action_id=action, action_name=name, description=desc)

@app.get("/")
def health_check():
    return {"status": "active", "service": "AI Tutor RL Agent"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
