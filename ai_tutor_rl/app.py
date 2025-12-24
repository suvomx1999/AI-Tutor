import streamlit as st
import numpy as np
import time
import matplotlib.pyplot as plt
import torch
import sys
import os

# Add current directory to path so imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.env import StudentEnv
from src.agent import DQNAgent

st.set_page_config(page_title="AI Tutor RL Dashboard", layout="wide")

st.title("🤖 AI Tutor Reinforcement Learning Dashboard")

st.sidebar.header("Simulation Parameters")
num_topics = st.sidebar.slider("Number of Topics", 3, 10, 5)
speed = st.sidebar.slider("Simulation Speed (sec/step)", 0.01, 1.0, 0.1, help="Lower is faster")

if st.sidebar.button("Run Simulation"):
    # Initialize Environment and Agent
    env = StudentEnv(num_topics=num_topics)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DQNAgent(state_dim, action_dim)
    try:
        agent.load('models/dqn_tutor.pth')
        st.success("Loaded trained DQN model.")
    except:
        st.warning("Model not found, using random agent.")

    # Containers for plots
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Live Student Stats")
        stats_placeholder = st.empty()
    with col2:
        st.subheader("Learning Progress")
        chart_placeholder = st.empty()
        
    # Run Episode
    state, _ = env.reset()
    done = False
    truncated = False
    
    steps = []
    difficulties = []
    scores = []
    knowledge_levels = []
    
    step_count = 0
    
    while not (done or truncated):
        action = agent.get_action(state, eval_mode=True)
        next_state, reward, done, truncated, info = env.step(action)
        
        # Data Collection
        steps.append(step_count)
        difficulties.append(env.current_difficulty)
        scores.append(env.last_score)
        knowledge_levels.append(np.mean(info['knowledge']))
        
        # Display Stats
        with stats_placeholder.container():
            st.metric("Current Topic", f"Topic {env.current_topic}")
            st.metric("Difficulty", f"{env.current_difficulty:.2f}")
            st.metric("Engagement", f"{info['engagement']:.2f}")
            st.progress(min(1.0, max(0.0, info['engagement'])))
            
            st.write("Topic Knowledge:")
            st.bar_chart(info['knowledge'])

        # Update Charts
        with chart_placeholder.container():
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(steps, difficulties, label='Difficulty', color='orange')
            ax.plot(steps, knowledge_levels, label='Avg Knowledge', color='blue')
            ax.set_ylim(0, 1.1)
            ax.legend()
            st.pyplot(fig)

        state = next_state
        step_count += 1
        time.sleep(speed)
        
    st.success("Simulation Complete!")
