# AI Tutor Web Client

This is a simple frontend to interact with the AI Tutor API.

## How to Run

1. **Start the API Server** (if not already running):
   ```bash
   cd ..
   uvicorn api:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Open the Client**:
   Simply open `index.html` in your web browser.
   
   On macOS:
   ```bash
   open index.html
   ```

## Features
- Displays current Topic and Difficulty.
- Allows you to simulate a quiz score (0-10).
- Sends your state to the RL Agent.
- Updates the difficulty/topic based on the Agent's recommendation.
