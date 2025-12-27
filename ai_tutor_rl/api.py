from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np
import os
import sys

# Add current directory to path so 'src' module can be found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agent import DQNAgent
from src.nlp_engine import NLPEngine

app = FastAPI(title="AI Tutor RL API", description="Real-time RL-based tutoring recommendations for integration with LMS.")

# Add CORS Middleware to allow requests from the browser client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load Model
NUM_TOPICS = 10
STATE_DIM = 6
ACTION_DIM = 5
HIDDEN_DIM = 128

agent = DQNAgent(STATE_DIM, ACTION_DIM, HIDDEN_DIM)
MODEL_PATH = "models/dqn_tutor.pth"

# Load NLP Engine
try:
    nlp_engine = NLPEngine()
except Exception as e:
    print(f"⚠️ Error loading NLP Engine: {e}")
    nlp_engine = None

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

# --- NLP Interaction Endpoints ---

class InteractionRequest(BaseModel):
    state: StudentState
    user_answer: str
    reference_answer: str

class InteractionResponse(BaseModel):
    grade: float
    feedback_text: str
    next_question: str
    next_reference: str
    new_state: StudentState
    action_name: str

@app.get("/start_session", response_model=InteractionResponse)
def start_session():
    """Initialize a new student session."""
    # Default initial state
    initial_state = StudentState(
        current_topic=0,
        current_difficulty=0.5,
        last_score=0.0,
        last_time=0.0,
        consecutive_failures=0,
        engagement=1.0
    )
    
    # Get first question
    q_text, a_text = nlp_engine.get_question(0, 0.5)
    
    return InteractionResponse(
        grade=0.0,
        feedback_text="Welcome! Let's start with Python Basics.",
        next_question=q_text,
        next_reference=a_text,
        new_state=initial_state,
        action_name="Start"
    )

@app.post("/submit_answer", response_model=InteractionResponse)
def submit_answer(request: InteractionRequest):
    """
    1. Grade user answer (NLP)
    2. Update Student State
    3. Get RL Action
    4. Fetch Next Question
    """
    if nlp_engine is None:
        raise HTTPException(status_code=500, detail="NLP Engine not loaded")
        
    # 1. Grade Answer
    grade = nlp_engine.grade_answer(request.user_answer, request.reference_answer)
    
    # 2. Update State
    state = request.state
    state.last_score = grade
    
    # Simple logic for failures/engagement update
    if grade < 4.0:
        state.consecutive_failures += 1
        state.engagement = max(0.0, state.engagement - 0.1)
    else:
        state.consecutive_failures = 0
        state.engagement = min(1.0, state.engagement + 0.1)
        
    # 3. Get Recommendation (Reuse logic by calling internal function or just copy logic)
    # We'll just call the get_recommendation function logic directly or via internal helper.
    # Since get_recommendation is an endpoint, calling it directly works if it returns the object.
    rec = get_recommendation(state)
    
    # 4. Apply Action to State (Transition Function)
    if rec.action_id == 0: # Easier
        state.current_difficulty = max(0.1, round(state.current_difficulty - 0.1, 2))
    elif rec.action_id == 1: # Harder
        state.current_difficulty = min(1.0, round(state.current_difficulty + 0.1, 2))
    elif rec.action_id == 2: # Revision
        # Keep same topic, maybe lower difficulty slightly or keep same
        pass
    elif rec.action_id == 4: # Next Topic
        state.current_topic = (state.current_topic + 1) % NUM_TOPICS
        state.current_difficulty = 0.5
        state.consecutive_failures = 0
        
    # 5. Fetch Next Question
    q_text, a_text = nlp_engine.get_question(
        state.current_topic, 
        state.current_difficulty,
        exclude_answer=request.reference_answer
    )
    
    return InteractionResponse(
        grade=grade,
        feedback_text=f"Score: {grade}/10. {rec.description}",
        next_question=q_text,
        next_reference=a_text,
        new_state=state,
        action_name=rec.action_name
    )

@app.get("/")
def health_check():
    return {"status": "active", "service": "AI Tutor RL Agent"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
