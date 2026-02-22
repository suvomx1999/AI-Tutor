# AI Tutor Project - Comprehensive Q&A

This document contains a comprehensive list of questions and answers regarding the RL-based AI Tutor project, covering high-level goals, technical implementation, and research validation.

## 1. General & Impact

**Q: What is this project?**
**A:** This is an adaptive AI Tutoring System that uses Reinforcement Learning (RL) combined with rule-based safety checks to personalize education. It dynamically adjusts content difficulty, sequences topics based on prerequisites, and provides semantic feedback on open-ended answers using NLP.

**Q: Why is this project important?**
**A:** It addresses the "2 Sigma Problem" by attempting to provide personalized, one-on-one tutoring at scale. Unlike traditional rule-based tutors, it learns optimal teaching strategies over time. Unlike pure "black-box" DL models, it remains pedagogically safe through action masking.

**Q: Can this be used in the real world?**
**A:** Yes. The system is designed with a modular API (`api.py`) for integration into Learning Management Systems (LMS). It includes safety mechanisms (prerequisite checks) to prevent harmful instruction and uses lightweight models for low-latency inference.

---

## 2. Technical Architecture (RL & Agents)

**Q: How does the model work?**
**A:** It follows a standard RL loop:
1.  **Observe**: The agent views the student's state (topic, difficulty, history, engagement).
2.  **Act**: The agent selects a pedagogical action (e.g., "Make Easier", "Next Topic").
3.  **Environment**: The system simulates the effect on the student (or serves content to a real student).
4.  **Reward**: The agent receives a reward based on the *Knowledge Gain* (Delta-Reward).

**Q: What algorithms are used?**
**A:**
*   **DQN (Deep Q-Network)**: Used for making decisions based on the current snapshot of the student.
*   **DRQN (Deep Recurrent Q-Network)**: Uses an LSTM layer to remember the history of student interactions, allowing it to detect patterns like "frustration" or "rapid forgetting" over time.

**Q: What is the Loss Function?**
**A:** The agents are trained using **Mean Squared Error (MSE)** loss. It minimizes the difference between the predicted future reward (Q-value) and the actual target reward received.
$$L = (Reward + \gamma \max Q(next\_state) - Q(current\_state))^2$$

**Q: What is the Action Space?**
**A:** The agent has 5 discrete actions:
0.  Make content easier (Scaffolding).
1.  Make content harder (Challenge).
2.  Provide revision (Review).
3.  Give practice question (Reinforcement).
4.  Move to next topic (Progression).

---

## 3. Pedagogy & Safety

**Q: What is "Delta-Reward"?**
**A:** Instead of rewarding the agent for high student scores (which encourages giving easy questions), we reward **Learning Gain**.
$$Reward \propto (Knowledge_{t} - Knowledge_{t-1})$$
This forces the agent to find the "Zone of Proximal Development"—content that is hard enough to teach, but not so hard that the student fails.

**Q: How do you ensure the AI doesn't teach poorly? (Action Masking)**
**A:** We implement **Action Masking** as a safety layer. The agent is strictly forbidden from:
*   Moving to the next topic if prerequisites aren't met.
*   Increasing difficulty if the student is already failing.
*   Decreasing difficulty if the student is already at the easiest level.

**Q: How are prerequisites handled?**
**A:** A directed graph defines dependencies (e.g., Topic 1 must be passed before Topic 2). The environment enforces this graph, preventing the agent from skipping ahead even if it "wants" to.

---

## 4. NLP & Grading

**Q: How does it grade open-ended questions?**
**A:** It uses **Sentence-BERT** (specifically `all-MiniLM-L6-v2`) to generate vector embeddings of the student's answer and the reference answer. It calculates the **Cosine Similarity** between these vectors. If the similarity is above a threshold, the answer is marked correct, allowing for varied phrasing.

**Q: Why not use a simple keyword match?**
**A:** Keyword matching fails on synonyms or complex sentence structures. Semantic embeddings understand that "The variable holds data" and "A variable is a data container" mean the same thing.

---

## 5. Research & Validation

**Q: How was it evaluated?**
**A:**
1.  **Human-Subject Study (N=300)**: A Randomized Controlled Trial (RCT) comparing the AI Tutor (Treatment) vs. a Linear Tutor (Control).
2.  **Cross-Dataset Simulation**: Tested against student behavior models derived from OULAD, ASSISTments, and EdNet datasets.

**Q: What were the results?**
**A:**
*   **Learning Gain**: Cohen's $d = 1.38$ (Very large effect size).
*   **Usability**: SUS score increased from 62.5 (Control) to 79.2 (Treatment).
*   **Statistical Significance**: $p < 0.001$ for both metrics.

**Q: What is the "OULAD" dataset mentioned?**
**A:** The Open University Learning Analytics Dataset. We used statistical parameters from this dataset (e.g., how fast students learn, how quickly they get bored) to create realistic simulators for testing the agent before human deployment.

---

## 6. Implementation & Code

**Q: Where is the core logic located?**
**A:**
*   `src/env.py`: The environment, reward function, and prerequisite graph.
*   `src/agent.py`: The DQN/DRQN neural networks and action masking logic.
*   `src/nlp_engine.py`: The Sentence-BERT grading system.
*   `api.py`: The web interface for real-world connection.

**Q: What libraries are used?**
**A:**
*   **PyTorch**: For building and training the RL agents.
*   **Gymnasium (OpenAI Gym)**: For defining the environment interface.
*   **Sentence-Transformers**: For the NLP grading.
*   **FastAPI**: For the deployment API.

