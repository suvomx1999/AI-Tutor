import numpy as np

class StudentSimulator:
    def __init__(self, num_topics=5):
        self.num_topics = num_topics
        # Initialize knowledge levels for each topic (0.0 to 1.0)
        # Some random initialization to simulate different students
        self.knowledge = np.random.uniform(0.1, 0.4, size=num_topics)
        self.engagement = 1.0  # Starts fully engaged
        self.fatigue = 0.0
        
    def attempt_question(self, topic_id, difficulty):
        """
        Simulate a student attempting a question.
        Returns: is_correct (bool), score (float), time_taken (float)
        """
        topic_knowledge = self.knowledge[topic_id]
        
        # Probability of success depends on knowledge vs difficulty
        # Sigmoid-like probability
        # difficulty is 0.0 to 1.0
        
        success_prob = 1 / (1 + np.exp(5 * (difficulty - topic_knowledge)))
        
        # Adjust by engagement (lower engagement -> lower success chance due to carelessness)
        success_prob *= self.engagement
        
        is_correct = np.random.random() < success_prob
        
        # Score based on difficulty and correctness
        # score = (difficulty * 10) if is_correct else 0
        # FIXED: Score is now continuous (0-10) to simulate semantic grading
        if is_correct:
            # Correct answers get high scores (7.5 - 10.0)
            score = np.random.uniform(7.5, 10.0)
        else:
            # Incorrect answers get lower scores (0.0 - 7.0)
            # Higher knowledge might mean a "better" wrong answer (partial credit)
            max_wrong_score = 4.0 + (3.0 * topic_knowledge) # Up to 7.0
            score = np.random.uniform(0.0, min(7.0, max_wrong_score))
            
        score = round(score, 1)
        
        # Time taken: harder questions take longer. Higher knowledge reduces time.
        base_time = 20  # seconds (Reduced from 30 to make students faster)
        time_taken = base_time * (1 + difficulty) * (1 - topic_knowledge * 0.5)
        # Add some noise
        time_taken += np.random.normal(0, 5)
        time_taken = max(5, time_taken)
        
        # Update internal state
        self._update_state(is_correct, difficulty, topic_id)
        
        return is_correct, score, time_taken

    def _update_state(self, is_correct, difficulty, topic_id):
        """
        Update student's knowledge and engagement based on the attempt.
        """
        # Learning: if correct and difficult, learn more. If incorrect, learn from mistake (less).
        learning_rate = 0.25 # Increased from 0.15 to accelerate learning
        if is_correct:
            # Learning gain proportional to difficulty
            gain = learning_rate * difficulty * self.engagement
            self.knowledge[topic_id] = min(1.0, self.knowledge[topic_id] + gain)
            # Success increases engagement
            self.engagement = min(1.0, self.engagement + 0.05)
        else:
            # Still learn a bit from failure, but less
            gain = learning_rate * 0.1
            self.knowledge[topic_id] = min(1.0, self.knowledge[topic_id] + gain)
            # Failure decreases engagement
            self.engagement = max(0.0, self.engagement - 0.05)
            
        # Fatigue increases with every attempt (Reduced for longer sessions)
        self.fatigue += 0.001
        # High fatigue reduces engagement
        if self.fatigue > 0.8:
            self.engagement = max(0.0, self.engagement - 0.1)

    def study(self, topic_id, intensity):
        """
        Simulate studying/revision (Action: Provide revision material)
        """
        gain = 0.1 * intensity * self.engagement
        self.knowledge[topic_id] = min(1.0, self.knowledge[topic_id] + gain)
        self.fatigue += 0.01
