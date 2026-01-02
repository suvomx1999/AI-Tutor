import gymnasium as gym
from gymnasium import spaces
import numpy as np
from src.student import StudentSimulator

class StudentEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    Represents the interaction between the AI Tutor and a Student.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, num_topics=10):
        super(StudentEnv, self).__init__()
        self.num_topics = num_topics
        self.student = None
        
        # Prerequisites Graph
        # Linear Dependency: Topic N requires Topic N-1 to be mastered (>0.8)
        self.prerequisites = {i: [i-1] for i in range(1, num_topics)}
        
        # Actions:
        # 0: Recommend easier content
        # 1: Recommend harder content
        # 2: Provide revision material
        # 3: Give practice questions (same difficulty)
        # 4: Move to next topic
        self.action_space = spaces.Discrete(5)

        # Observation Space:
        # [current_topic_id, current_difficulty, last_score, last_time_taken, consecutive_failures, engagement_proxy]
        # We normalize values roughly to [0, 1] or use standard scaler later, but for now raw values.
        # topic_id: 0 to num_topics-1
        # difficulty: 0.0 to 1.0
        # score: 0 to 10
        # time: 0 to 300 (clip)
        # failures: 0 to 10
        # engagement: 0 to 1
        
        low = np.array([0, 0.0, 0, 0, 0, 0], dtype=np.float32)
        high = np.array([num_topics, 1.0, 10.0, 300.0, 20.0, 1.0], dtype=np.float32)
        
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.max_steps = 1000
        self.current_step = 0
        
        # Session State
        self.current_topic = 0
        self.current_difficulty = 0.5
        self.last_score = 0
        self.last_time = 0
        self.consecutive_failures = 0
        
        self.history = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.student = StudentSimulator(self.num_topics)
        self.current_step = 0
        self.current_topic = 0
        self.current_difficulty = 0.5 # Start with medium difficulty
        self.last_score = 0
        self.last_time = 0
        self.consecutive_failures = 0
        self.history = []
        
        observation = self._get_obs()
        return observation, {}

    def step(self, action):
        self.current_step += 1
        reward = 0
        done = False
        truncated = False
        info = {}

        prev_topic = self.current_topic
        prev_knowledge = self.student.knowledge[self.current_topic]
        
        # Execute Action
        # HARD OVERRIDE: If knowledge is > 0.95, FORCE action 4 (Next Topic)
        # This applies to ALL topics (0 to N-1), ensuring the student progresses
        # immediately upon mastery (95%).
        # STRICT PROGRESSION: Require high knowledge (>0.92) before forcing move
        if prev_knowledge > 0.92 and self.current_topic < self.num_topics - 1:
            action = 4
            
        executed_action = action

        if action == 0: # Easier
            self.current_difficulty = max(0.1, self.current_difficulty - 0.2)
            is_correct, score, time = self.student.attempt_question(self.current_topic, self.current_difficulty)
        
        elif action == 1: # Harder
            self.current_difficulty = min(1.0, self.current_difficulty + 0.2)
            is_correct, score, time = self.student.attempt_question(self.current_topic, self.current_difficulty)
            
        elif action == 2: # Revision
            # Revision doesn't give a score immediately, but improves knowledge
            # We simulate a small "quiz" after revision to get a state update
            self.student.study(self.current_topic, intensity=1.0)
            # Revision takes time
            time = 60 
            # FIXED: Return a score reflecting current mastery instead of 0
            score = self.student.knowledge[self.current_topic] * 10.0
            is_correct = True # Assume revision is "completed"
            
        elif action == 3: # Practice (Same difficulty)
            is_correct, score, time = self.student.attempt_question(self.current_topic, self.current_difficulty)
            
        elif action == 4: # Next Topic
            # Curriculum Enforcement
            # Check if prerequisites are met for the NEXT topic
            next_topic = self.current_topic + 1
            can_advance = True
            
            if next_topic < self.num_topics:
                reqs = self.prerequisites.get(next_topic, [])
                for req_id in reqs:
                    if self.student.knowledge[req_id] < 0.8: # Require 80% mastery of prerequisite
                        can_advance = False
                        break
            
            # Only allow moving if knowledge is sufficient (e.g. > 0.85) and prereqs met
            # Otherwise, treat as practice and give penalty
            # STRICT: Require >0.92 knowledge to voluntarily move
            if can_advance and self.student.knowledge[self.current_topic] > 0.92:
                if self.current_topic < self.num_topics - 1:
                    self.current_topic += 1
                    self.current_difficulty = 0.5 # Reset difficulty for new topic
                    self.consecutive_failures = 0
                    # Simulate a first probe question
                    is_correct, score, time = self.student.attempt_question(self.current_topic, self.current_difficulty)
                else:
                    # Already at last topic, treat as practice
                    is_correct, score, time = self.student.attempt_question(self.current_topic, self.current_difficulty)
            else:
                # Failed attempt to move (knowledge too low or prereqs missing)
                # Treat as practice but penalize
                is_correct, score, time = self.student.attempt_question(self.current_topic, self.current_difficulty)
                reward -= 20 # Penalty for trying to skip


        # Force progression mechanism (Optional safeguard)
        # If knowledge is extremely high, we override the agent's choice if it's dumb
        # But in RL philosophy we should let it learn. 
        # However, for user satisfaction, let's add a "Guidance Reward" for taking action 4 when ready.
        
        # Update State Variables
        self.last_score = score
        self.last_time = time
        if not is_correct and action != 2: # Don't count revision as failure
            self.consecutive_failures += 1
        else:
            self.consecutive_failures = 0

        # Calculate Reward (ENHANCED DELTA LOGIC)
        # 1. Base: Score (scaled 0-1)
        reward = self.last_score / 10.0
        
        # 2. Delta Reward: Improvement over previous attempt
        # We need to track previous score for delta
        # Since env doesn't track per-question history perfectly, we use knowledge gain proxy
        current_knowledge = self.student.knowledge[self.current_topic]
        knowledge_gain = current_knowledge - prev_knowledge
        
        # Huge reward for learning (moving the needle)
        if knowledge_gain > 0:
            reward += (knowledge_gain * 100.0) # e.g., 0.05 gain -> +5 reward
            
        # 3. Action Specific Rewards/Penalties
        
        # Penalize reducing difficulty if student is doing well (Boredom prevention)
        if action == 0 and prev_knowledge > 0.7:
            reward -= 0.5 
        
        # Reward increasing difficulty if student is cruising (Flow state)
        if action == 1 and prev_knowledge > 0.8:
            reward += 0.5
            
        # Penalize repeated failures (Frustration prevention)
        if self.consecutive_failures > 2:
            reward -= 1.0
            
        # Reward appropriate advancement (Curriculum pacing)
        if action == 4 and prev_topic != self.current_topic:
             reward += 5.0 # Big bonus for completing a topic
            
        if self.student.engagement < 0.2:
            reward -= 2.0
            done = True # End session if student is disengaged
            
        # Time penalty (efficiency) - small
        reward -= 0.01

        # Termination
        if self.current_step >= self.max_steps:
            truncated = True
        
        # Check if mastered all topics
        # Requirement: ALL topics must be mastered (> 0.9), not just the average.
        # STRICT: Require >0.92 minimum knowledge across ALL topics
        if np.min(self.student.knowledge) > 0.92:
            reward += 100
            done = True
        
        # HUGE Bonus for moving to next topic when ready
        if prev_knowledge > 0.9 and action == 4 and self.current_topic > prev_topic:
            reward += 50

        # Penalize sticking to a mastered topic too long
        # If knowledge in current topic is very high (>0.95), and we are not moving (Action 4), give penalty
        if self.student.knowledge[self.current_topic] > 0.95 and action != 4:
            if self.current_topic < self.num_topics - 1:
                reward -= 50 # Massive penalty for stagnating

        observation = self._get_obs()
        
        info = {
            'knowledge': self.student.knowledge.copy(),
            'engagement': self.student.engagement,
            'topic': self.current_topic,
            'executed_action': executed_action
        }
        self.history.append((action, reward, info))
        
        return observation, reward, done, truncated, info

    def _get_obs(self):
        return np.array([
            self.current_topic,
            self.current_difficulty,
            self.last_score,
            self.last_time,
            self.consecutive_failures,
            self.student.engagement
        ], dtype=np.float32)

    def render(self, mode='human'):
        print(f"Step: {self.current_step}")
        print(f"Topic: {self.current_topic}, Diff: {self.current_difficulty:.2f}")
        print(f"Knowledge: {self.student.knowledge}")
        print(f"Engagement: {self.student.engagement:.2f}")
