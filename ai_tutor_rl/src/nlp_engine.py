import numpy as np
import re
from sentence_transformers import SentenceTransformer, util

class NLPEngine:
    def __init__(self):
        # Load a lightweight model for semantic similarity
        print("⏳ Loading NLP Model (all-MiniLM-L6-v2)...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ NLP Model Loaded!")
        
        # Knowledge Base: Topic ID -> List of Questions
        # Each question has: text, difficulty (0.0-1.0), reference_answer
        self.content_db = {
            0: [ # Topic 0: Python Basics
                {"q": "What is a variable in Python?", "d": 0.1, "a": "A variable is a container for storing data values."},
                {"q": "How do you output text to the console in Python?", "d": 0.2, "a": "Use the print() function."},
                {"q": "How do you write a comment in Python?", "d": 0.3, "a": "Start the line with a hash symbol (#)."},
                {"q": "What data type is used for True/False values?", "d": 0.4, "a": "The Boolean type (bool)."},
                {"q": "What is the difference between an integer and a float?", "d": 0.5, "a": "An integer is a whole number, while a float is a number with a decimal point."},
                {"q": "How do you convert a string '123' to an integer?", "d": 0.6, "a": "Use the int() function: int('123')."},
                {"q": "What is the result of 10 // 3 in Python?", "d": 0.7, "a": "It is 3 (floor division)."},
                {"q": "Explain dynamic typing in Python.", "d": 0.8, "a": "In dynamic typing, variable types are determined at runtime, not compile time. You do not need to declare types explicitly, and a variable can change its type during execution."},
                {"q": "What is the 'None' keyword used for?", "d": 0.9, "a": "It represents a null value or no value at all."},
            ],
            1: [ # Topic 1: Control Flow
                {"q": "What keyword is used for a conditional statement?", "d": 0.2, "a": "The if keyword."},
                {"q": "How do you start a for loop that ranges from 0 to 4?", "d": 0.4, "a": "for i in range(5):"},
                {"q": "What is the purpose of the 'break' statement?", "d": 0.6, "a": "It terminates the loop immediately."},
                {"q": "Explain the difference between 'break' and 'continue'.", "d": 0.9, "a": "Break stops the loop completely, continue skips the current iteration and moves to the next one."},
            ],
            2: [ # Topic 2: Functions
                {"q": "How do you define a function in Python?", "d": 0.2, "a": "Use the def keyword followed by the function name."},
                {"q": "What is a return statement?", "d": 0.4, "a": "It sends a result back from a function to the caller."},
                {"q": "What are default arguments?", "d": 0.7, "a": "Parameters that assume a default value if no value is provided in the function call."},
                {"q": "Explain lambda functions.", "d": 0.9, "a": "Small anonymous functions defined with the lambda keyword, usually for short operations."},
            ]
        }
        
        # Default content for other topics to prevent errors
        for i in range(3, 10):
            self.content_db[i] = [
                {"q": f"Placeholder question for Topic {i} (Easy)", "d": 0.2, "a": "Answer"},
                {"q": f"Placeholder question for Topic {i} (Hard)", "d": 0.8, "a": "Answer"},
            ]

    def get_question(self, topic_id: int, difficulty: float, exclude_answer: str = None):
        """
        Returns the question closest to the requested difficulty for the given topic.
        Optionally excludes a specific question (identified by its answer) to prevent repetition.
        """
        questions = self.content_db.get(topic_id, [])
        if not questions:
            return "No questions available for this topic.", "N/A"
            
        # Filter out the excluded question if provided
        candidates = questions
        if exclude_answer:
            candidates = [q for q in questions if q['a'] != exclude_answer]
            
        # If filtering removed all questions (e.g. only 1 question exists), fall back to all questions
        if not candidates:
            candidates = questions

        # Find question with closest difficulty among candidates
        best_match = min(candidates, key=lambda x: abs(x['d'] - difficulty))
        return best_match['q'], best_match['a']

    def grade_answer(self, user_answer: str, reference_answer: str) -> float:
        """
        Computes semantic similarity between user answer and reference.
        Returns a score between 0.0 and 10.0
        """
        if not user_answer.strip():
            return 0.0
            
        user_clean = user_answer.strip().lower()
        ref_clean = reference_answer.strip().lower()

        # 1. Exact match (case insensitive)
        if user_clean == ref_clean:
            return 10.0
            
        # 2. Numeric match override
        # If user answer is just a number (e.g. "3" or "3.0")
        if re.match(r'^-?\d+(\.\d+)?$', user_clean):
            # Check if this number appears as a word in reference answer
            # We use regex to ensure we match "3" but not "30" or "13"
            # escape user_clean to handle potential regex special chars if any (though digits are safe)
            if re.search(r'\b' + re.escape(user_clean) + r'\b', ref_clean):
                return 10.0
            
        # 3. Semantic Similarity
        embeddings = self.model.encode([user_answer, reference_answer])
        
        # Compute cosine similarity
        sim = util.cos_sim(embeddings[0], embeddings[1]).item()
        
        # Scale to 0-10
        # Similarity is usually -1 to 1, but for text it's usually 0 to 1.
        score = max(0.0, sim) * 10.0
        
        # 4. Generous Grading Curve
        # Raw cosine similarity is often harsh (0.6 is actually a good match).
        # We boost scores that are "decent" (above 4.0) to encourage users.
        if score > 4.0:
            # Boost range: 4.0 -> 10.0
            # Formula: 4.0 + (score - 4.0) * 1.8
            # Example: Raw 6.0 -> 4.0 + 3.6 = 7.6
            # Example: Raw 7.0 -> 4.0 + 5.4 = 9.4
            score = 4.0 + (score - 4.0) * 1.8
            
        # Cap at 10.0
        score = min(10.0, score)
        
        return round(score, 2)
