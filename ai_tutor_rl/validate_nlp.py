import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.nlp_engine import NLPEngine
import numpy as np
from scipy.stats import pearsonr, spearmanr

def validate_nlp():
    print("Initializing NLP Engine...")
    nlp = NLPEngine()
    
    # Synthetic Validation Dataset
    # Format: (Question, User Answer, Reference Answer, Human Score 0-10)
    dataset = [
        # Exact/Near Exact Matches (High Score)
        ("What is a variable?", "A variable is a container for storing data values.", "A variable is a container for storing data values.", 10.0),
        ("What is a variable?", "A variable is essentially a container to store data.", "A variable is a container for storing data values.", 9.5),
        ("How do you output text?", "Use the print() function.", "Use the print() function.", 10.0),
        ("How do you output text?", "You can use print() to show text on console.", "Use the print() function.", 9.0),
        
        # Paraphrases (High/Medium Score)
        ("What is a variable?", "It's a box where we put values.", "A variable is a container for storing data values.", 8.0),
        ("What is a variable?", "Something that holds information.", "A variable is a container for storing data values.", 7.5),
        ("How do you output text?", "print('text')", "Use the print() function.", 8.5),
        
        # Partial Correctness (Medium/Low Score)
        ("What is a variable?", "It is used in coding.", "A variable is a container for storing data values.", 4.0),
        ("How do you output text?", "Use output() function.", "Use the print() function.", 3.0), # Wrong function name but right idea
        ("What is a list?", "It uses brackets.", "A list is a mutable sequence.", 3.5),
        
        # Irrelevant/Wrong (Low Score)
        ("What is a variable?", "I don't know.", "A variable is a container for storing data values.", 0.0),
        ("What is a variable?", "Pizza is tasty.", "A variable is a container for storing data values.", 0.0),
        ("How do you output text?", "x = 5", "Use the print() function.", 0.0),
        
        # Adversarial / Keyword Stuffing (Tricky)
        # SBERT might overscore these, but let's see. Human score is low.
        ("What is a variable?", "variable container data values store.", "A variable is a container for storing data values.", 2.0),
        ("How do you output text?", "print function console text.", "Use the print() function.", 2.0),
    ]
    
    human_scores = []
    system_scores = []
    
    print(f"\nEvaluating {len(dataset)} samples...")
    print(f"{'User Answer':<40} | {'Ref Answer':<40} | {'Human':<5} | {'System':<5}")
    print("-" * 100)
    
    for q, user_a, ref_a, h_score in dataset:
        s_score = nlp.grade_answer(user_a, ref_a)
        
        human_scores.append(h_score)
        system_scores.append(s_score)
        
        print(f"{user_a[:37]:<40} | {ref_a[:37]:<40} | {h_score:<5.1f} | {s_score:<5.1f}")
        
    human_scores = np.array(human_scores)
    system_scores = np.array(system_scores)
    
    # Calculate Correlation
    p_corr, p_val = pearsonr(human_scores, system_scores)
    s_corr, s_val = spearmanr(human_scores, system_scores)
    
    print("\n" + "="*50)
    print("NLP Validation Results")
    print("="*50)
    print(f"Pearson Correlation (r):  {p_corr:.4f} (p={p_val:.4e})")
    print(f"Spearman Correlation (rho): {s_corr:.4f} (p={s_val:.4e})")
    print("="*50)
    
    # Generate LaTeX Table Snippet
    print("\nLaTeX Table Snippet:")
    print(r"\begin{table}[htbp]")
    print(r"\caption{NLP Validation Results}")
    print(r"\centering")
    print(r"\begin{tabular}{|l|c|}")
    print(r"\hline")
    print(r"\textbf{Metric} & \textbf{Value} \\")
    print(r"\hline")
    print(f"Sample Size & {len(dataset)} \\\\")
    print(f"Pearson $r$ & {p_corr:.2f} \\\\")
    print(f"Spearman $\\rho$ & {s_corr:.2f} \\\\")
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\label{tab:nlp_val}")
    print(r"\end{table}")

if __name__ == "__main__":
    validate_nlp()
