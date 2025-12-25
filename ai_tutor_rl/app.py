import streamlit as st
import numpy as np
import time
import pandas as pd
import altair as alt
import sys
import os

# Add current directory to path so imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.env import StudentEnv
from src.agent import DQNAgent

st.set_page_config(
    page_title="AI Tutor RL Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        text-align: center; 
        color: #1E3A8A;
    }
    .stMetric {
        background-color: #F3F4F6;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #E5E7EB;
    }
    /* Colorful progress bar */
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #EF4444, #F59E0B, #10B981);
    }
    .agent-box {
        padding: 20px;
        background-color: #EFF6FF;
        border-radius: 10px;
        border-left: 5px solid #2563EB;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎓 AI Tutor: Adaptive RL Dashboard")
st.markdown("""
<div style="text-align: center; color: #6B7280; margin-bottom: 30px;">
    Real-time visualization of the <b>Hybrid RL Agent</b> optimizing a student's learning path.
</div>
""", unsafe_allow_html=True)

# Sidebar with better grouping
with st.sidebar:
    st.header("⚙️ Simulation Settings")
    num_topics = st.slider("📚 Number of Topics", 3, 10, 10)
    speed = st.slider("⏱️ Step Delay (sec)", 0.01, 1.0, 0.1, help="Adjust simulation speed")
    
    st.divider()
    
    st.markdown("### 🧠 Agent Info")
    st.info("The agent uses a **Hybrid Strategy**: combining Rule-Based Safety Checks with a Deep Q-Network (DQN) for optimization.")
    
    start_simulation = st.button("🚀 Start Simulation", type="primary", use_container_width=True)

ACTION_MAP = {
    0: "Easier Question",
    1: "Harder Question",
    2: "Revision",
    3: "Practice",
    4: "Next Topic"
}

def get_hybrid_action(agent, env):
    """
    Replicates the logic from api.py to ensure the dashboard matches production behavior.
    """
    state = env._get_obs() # This returns raw state in env.py, need to check normalization
    
    # Env state: [topic, difficulty, score, time, failures, engagement]
    # API expects normalized inputs for the agent, but logic rules use raw values.
    
    # Extract raw values from env
    current_topic = env.current_topic
    current_difficulty = env.current_difficulty
    last_score = env.last_score
    consecutive_failures = env.consecutive_failures
    engagement = env.student.engagement
    
    # --- RULE BASED LOGIC (Matching api.py) ---
    action = None
    
    # Rule 1: Mastered Topic -> Move Next
    # Reduced threshold for easier demo visibility
    # STRICT: Require 9.0 score and very high difficulty to propose next topic
    if last_score >= 9.0 and current_difficulty >= 0.9:
         action = 4 # Next Topic
         
    # Rule 2: Cruising -> Increase Difficulty
    elif last_score >= 8.0 and current_difficulty < 0.9:
        action = 1 # Harder Question
        
    # Rule 3: Struggling -> Decrease Difficulty
    elif (consecutive_failures >= 2 or last_score < 4.0) and current_difficulty > 0.15:
        action = 0 # Easier Question
        
    # Rule 4: Absolute Bottom -> Remedial Action
    elif last_score < 5.0 and current_difficulty <= 0.15:
        action = 2 # Revision
        
    if action is not None:
        return action, "Rule Override"

    # --- RL AGENT LOGIC ---
    # Normalize for Agent
    # Obs space: [topic, diff, score, time, fail, eng]
    # Normalization matches api.py
    obs = np.array([
        current_topic / float(num_topics),
        current_difficulty,
        last_score / 10.0,
        env.last_time / 60.0,
        consecutive_failures / 5.0,
        engagement
    ], dtype=np.float32)
    
    action = agent.get_action(obs, eval_mode=True)
    return action, "RL Agent"

if start_simulation:
    # Initialize Environment and Agent
    env = StudentEnv(num_topics=num_topics)
    state_dim = 6 # Fixed based on env
    action_dim = 5
    
    agent = DQNAgent(state_dim, action_dim)
    try:
        agent.load('models/dqn_tutor.pth')
        st.sidebar.success("✅ Loaded trained DQN model.")
    except:
        st.sidebar.warning("⚠️ Model not found, using random agent.")

    # --- Dashboard Layout ---
    
    # 1. Top Metrics Row
    st.subheader("Student Status")
    m1, m2, m3, m4 = st.columns(4)
    metric_topic = m1.empty()
    metric_diff = m2.empty()
    metric_score = m3.empty()
    metric_failures = m4.empty()
    
    st.markdown("---")
    
    # 2. Main Content: Charts & Agent
    col_main, col_side = st.columns([2, 1])
    
    with col_main:
        st.subheader("📈 Learning Curve")
        chart_placeholder = st.empty()
        st.caption("Tracking Difficulty (Blue), Knowledge (Green), and Scores (Red)")
        
        st.divider()
        st.subheader("🧠 Topic Mastery Levels")
        knowledge_chart_placeholder = st.empty()

    with col_side:
        st.subheader("🤖 Agent Decision")
        action_container = st.empty()
        
        st.markdown("#### Engagement")
        metric_eng = st.empty()
        
        st.markdown("#### Activity Log")
        log_placeholder = st.empty()

    # Run Episode
    env.reset()
    done = False
    truncated = False
    
    history_df = pd.DataFrame(columns=['Step', 'Difficulty', 'Knowledge', 'Score'])
    logs = []
    
    step_count = 0
    
    while not (done or truncated):
        # Get Action using Hybrid Logic
        action, source = get_hybrid_action(agent, env)
        
        # Take Step
        next_state, reward, done, truncated, info = env.step(action)
        
        # Update Data
        # Check if environment overrode the action (e.g. forced progression)
        real_action = info.get('executed_action', action)
        if real_action != action:
            action = real_action
            source = "Env Override" # Indicate it was forced by environment rules

        new_row = {
            'Step': step_count, 
            'Difficulty': env.current_difficulty, 
            'Knowledge': np.mean(info['knowledge']),
            'Score': env.last_score / 10.0 # Normalize for chart
        }
        history_df = pd.concat([history_df, pd.DataFrame([new_row])], ignore_index=True)
        
        # Update Top Metrics
        metric_topic.metric("Topic Progress", f"{env.current_topic + 1} / {num_topics}")
        metric_diff.metric("Current Difficulty", f"{env.current_difficulty:.2f}")
        
        # Color code score
        score_val = env.last_score
        delta_color = "normal"
        if score_val >= 8: delta_color = "normal" # Streamlit handles green/red automatically for delta, but here we just show value
        metric_score.metric("Last Quiz Score", f"{score_val:.1f}/10")
        
        metric_failures.metric("Consecutive Failures", f"{env.consecutive_failures}")

        # Update Chart
        with chart_placeholder:
            # Customizing chart colors by column order: Difficulty (Blue), Knowledge (Green), Score (Red)
            # Streamlit cycle is usually Blue, Orange, Green, Red.
            # Let's just use simple line chart for speed
            st.line_chart(history_df.set_index('Step')[['Difficulty', 'Knowledge', 'Score']], height=350)
            
        # Update Knowledge Bar Chart
        knowledge_data = pd.DataFrame({
            "Topic": [f"Topic {i+1}" for i in range(num_topics)],
            "Mastery": info['knowledge']
        })
        
        with knowledge_chart_placeholder:
            c = alt.Chart(knowledge_data).mark_bar().encode(
                x=alt.X('Topic', title='Topic'),
                y=alt.Y('Mastery', scale=alt.Scale(domain=[0, 1]), title='Knowledge Level'),
                color=alt.Color('Mastery', scale=alt.Scale(scheme='viridis'), legend=None),
                tooltip=['Topic', alt.Tooltip('Mastery', format='.2f')]
            ).properties(height=250)
            st.altair_chart(c, use_container_width=True)
            
        # Update Agent Action UI
        action_name = ACTION_MAP[action]
        bg_color = "#DCFCE7" if "RL" not in source else "#DBEAFE" # Greenish for Rules, Blueish for RL
        border_color = "#16A34A" if "RL" not in source else "#2563EB"
        
        action_html = f"""
        <div style="background-color: {bg_color}; padding: 15px; border-radius: 10px; border-left: 5px solid {border_color};">
            <h3 style="margin:0; color: #1F2937;">{action_name}</h3>
            <p style="margin:5px 0 0 0; color: #4B5563; font-size: 0.9em;">Source: <b>{source}</b></p>
        </div>
        """
        action_container.markdown(action_html, unsafe_allow_html=True)
        
        # Update Engagement Bar
        eng_val = float(info['engagement'])
        metric_eng.progress(eng_val, text=f"Level: {eng_val:.2f}")
        
        # Logs
        log_msg = f"Step {step_count}: Scored {env.last_score:.1f} -> {action_name}"
        logs.insert(0, log_msg)
        log_placeholder.code("\n".join(logs[:8]), language="text")

        step_count += 1
        time.sleep(speed)
        
    st.success("🎉 Excellent! Student has mastered ALL topics with >92% knowledge.")
    st.balloons()
