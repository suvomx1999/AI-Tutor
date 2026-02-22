# Human Subject Study Protocol: Hybrid AI Tutor Evaluation

## 1. Study Objective
To evaluate the efficacy of the Hybrid RL-NLP Tutoring System compared to a linear curriculum baseline in a controlled educational setting.

## 2. Research Questions
- **RQ1 (Learning Gain):** Does the Hybrid RL agent lead to significantly higher normalized learning gains (Hake's g) compared to a linear playlist?
- **RQ2 (Engagement):** Does the personalized curriculum reduce self-reported boredom and frustration?
- **RQ3 (Usability):** Is the AI tutor perceived as helpful and adaptive by students?

## 3. Participants
- **Target N:** 60 participants (30 Control, 30 Treatment).
- **Recruitment:** Undergraduate CS students (novice to Python).
- **Inclusion Criteria:** No prior experience with the specific Python topics (verified via pre-screen).

## 4. Experimental Design
Randomized Controlled Trial (RCT) with two arms:
1.  **Control Group (Linear):** Students proceed through 5 topics in a fixed order (Video -> Quiz -> Next). No skipping, no remedial loops.
2.  **Treatment Group (Adaptive):** Students interact with the Hybrid RL Agent. The agent controls the sequence (Video, Quiz, Remedial, Skip).

## 5. Procedure (Duration: ~60 mins)
1.  **Onboarding (5 mins):** Sign Consent Form. Introduction to interface.
2.  **Pre-Test (10 mins):** 10 multiple-choice questions covering all 5 topics.
3.  **Learning Phase (35 mins):** Interaction with the system (Control or Treatment).
    - Max time cap: 35 mins.
4.  **Post-Test (10 mins):** 10 multiple-choice questions (isomorphic to Pre-Test).
5.  **Survey (5 mins):** NASA-TLX (Workload) + System Usability Scale (SUS) + Custom Engagement questions.

## 6. Metrics
- **Learning Gain:** $g = \frac{Post - Pre}{1 - Pre}$
- **Retention:** (Optional) Delayed post-test after 1 week.
- **Engagement:** Time on task, number of voluntary interactions, survey Likert scales.

## 7. Data Collection & Privacy
- All data is anonymized.
- IDs are hashed.
- Log files stored in secure JSON format: `study_data/{participant_id}_log.json`.
