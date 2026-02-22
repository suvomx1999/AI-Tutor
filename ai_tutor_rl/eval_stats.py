import os
import json
import numpy as np
import random
from typing import Dict, Tuple
from src.env import StudentEnv
from src.agent import DQNAgent, DRQNAgent, LSTMTutorPolicy
from src.utils import smart_static_policy

def evaluate_once(agent, episodes=30, seed=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    env = StudentEnv()
    rewards = []
    gains = []
    engagements = []
    steps = []
    scores_ep = []
    scores_post = []
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        total = 0.0
        init_k = np.mean(env.student.knowledge)
        ep_eng = []
        step_count = 0
        ep_scores = []
        while not (done or truncated):
            ep_eng.append(env.student.engagement)
            if agent is None:
                action = smart_static_policy(state, env.current_difficulty)
            else:
                try:
                    action = agent.get_action(state, eval_mode=True)
                except TypeError:
                    action = agent.get_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            state = next_state
            total += reward
            step_count += 1
            ep_scores.append(env.last_score)
        rewards.append(total)
        gains.append(np.mean(env.student.knowledge) - init_k)
        engagements.append(np.mean(ep_eng))
        steps.append(step_count)
        scores_ep.append(float(np.mean(ep_scores)) if len(ep_scores) > 0 else 0.0)
        # Standardized post-test (10 medium-difficulty probes)
        post_scores = []
        num_items = 10
        for i in range(num_items):
            # Focus half on current topic, half on previous topics if any
            if i < num_items // 2 or env.current_topic == 0:
                t_id = env.current_topic
            else:
                t_id = random.randint(0, env.current_topic)
            # Medium difficulty probe
            _, sc, _ = env.student.attempt_question(t_id, difficulty=0.5)
            post_scores.append(sc)
        scores_post.append(float(np.mean(post_scores)) if len(post_scores) > 0 else 0.0)
    return np.array(rewards), np.array(gains), np.array(engagements), np.array(steps), np.array(scores_ep), np.array(scores_post)

def mean_ci(x: np.ndarray) -> Tuple[float, float]:
    m = float(np.mean(x))
    s = float(np.std(x, ddof=1))
    n = len(x)
    ci = 1.96 * s / np.sqrt(n) if n > 1 else 0.0
    return m, ci

def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    mx, my = np.mean(x), np.mean(y)
    sx, sy = np.std(x, ddof=1), np.std(y, ddof=1)
    nx, ny = len(x), len(y)
    sp = np.sqrt(((nx - 1) * sx**2 + (ny - 1) * sy**2) / (nx + ny - 2)) if (nx + ny - 2) > 0 else 0.0
    if sp == 0.0:
        return 0.0
    return float((mx - my) / sp)

def permutation_pvalue(x: np.ndarray, y: np.ndarray, n_perm: int = 5000) -> float:
    obs = abs(np.mean(x) - np.mean(y))
    pooled = np.concatenate([x, y])
    nx = len(x)
    count = 0
    rng = np.random.default_rng(42)
    for _ in range(n_perm):
        rng.shuffle(pooled)
        x_s = pooled[:nx]
        y_s = pooled[nx:]
        diff = abs(np.mean(x_s) - np.mean(y_s))
        if diff >= obs:
            count += 1
    return (count + 1) / (n_perm + 1)

def main():
    env = StudentEnv()
    sdim = env.observation_space.shape[0]
    adim = env.action_space.n
    dqn = DQNAgent(sdim, adim)
    drqn = DRQNAgent(sdim, adim)
    lstm = LSTMTutorPolicy(sdim, adim, seq_len=12, hidden_dim=128)
    if os.path.exists('models/dqn_tutor.pth'):
        dqn.load('models/dqn_tutor.pth')
    if os.path.exists('models/drqn_tutor.pth'):
        drqn.load('models/drqn_tutor.pth')
    if os.path.exists('models/lstm_tutor.pth'):
        lstm.load('models/lstm_tutor.pth')
    labels = ['random', 'rule_based', 'dqn', 'drqn', 'lstm']
    agents = {
        'random': type('RandomPolicy', (), {'get_action': lambda self, state, eval_mode=True: random.randint(0, adim-1)})(),
        'rule_based': None,  # handled via smart_static_policy in evaluate_once
        'dqn': dqn,
        'drqn': drqn,
        'lstm': lstm
    }
    seeds = [11, 23, 37, 49, 61]
    ep_per_seed = 20
    dist_rewards: Dict[str, np.ndarray] = {}
    dist_gains: Dict[str, np.ndarray] = {}
    dist_eng: Dict[str, np.ndarray] = {}
    dist_steps: Dict[str, np.ndarray] = {}
    dist_scores_ep: Dict[str, np.ndarray] = {}
    dist_scores_post: Dict[str, np.ndarray] = {}
    for name in labels:
        all_r = []
        all_g = []
        all_e = []
        all_s = []
        all_sc_ep = []
        all_sc_post = []
        for s in seeds:
            r, g, e, st, sc_ep, sc_post = evaluate_once(agents[name] if name != 'rule_based' else None, episodes=ep_per_seed, seed=s)
            all_r.append(r)
            all_g.append(g)
            all_e.append(e)
            all_s.append(st)
            all_sc_ep.append(sc_ep)
            all_sc_post.append(sc_post)
        dist_rewards[name] = np.concatenate(all_r)
        dist_gains[name] = np.concatenate(all_g)
        dist_eng[name] = np.concatenate(all_e)
        dist_steps[name] = np.concatenate(all_s)
        dist_scores_ep[name] = np.concatenate(all_sc_ep)
        dist_scores_post[name] = np.concatenate(all_sc_post)
    
    # compute stats
    stats = {'rewards': {}, 'gains': {}, 'engagement': {}, 'steps': {}, 'scores_episode': {}, 'scores_posttest': {}}
    for name in labels:
        m, ci = mean_ci(dist_gains[name])
        stats['gains'][name] = {'mean': round(m, 4), 'ci95': round(ci, 4), 'n': int(len(dist_gains[name]))}
        m2, ci2 = mean_ci(dist_rewards[name])
        stats['rewards'][name] = {'mean': round(m2, 2), 'ci95': round(ci2, 2), 'n': int(len(dist_rewards[name]))}
        m3, ci3 = mean_ci(dist_eng[name])
        stats['engagement'][name] = {'mean': round(m3, 2), 'ci95': round(ci3, 2)}
        m4, ci4 = mean_ci(dist_steps[name])
        stats['steps'][name] = {'mean': round(m4, 1), 'ci95': round(ci4, 1)}
        m5, ci5 = mean_ci(dist_scores_ep[name])
        stats['scores_episode'][name] = {'mean': round(m5, 1), 'ci95': round(ci5, 1)}
        m6, ci6 = mean_ci(dist_scores_post[name])
        stats['scores_posttest'][name] = {'mean': round(m6, 1), 'ci95': round(ci6, 1)}

    # effect sizes vs rule_based (knowledge gain)
    for name in ['dqn', 'drqn', 'lstm']:
        d = cohen_d(dist_gains[name], dist_gains['rule_based'])
        stats['gains'][name]['cohen_d_vs_rule'] = round(d, 3)
        p = permutation_pvalue(dist_gains[name], dist_gains['rule_based'], n_perm=5000)
        stats['gains'][name]['p_value_vs_rule'] = round(p, 4)
    
    # save
    if not os.path.exists('plots'):
        os.makedirs('plots')
    with open('plots/paper_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))

if __name__ == '__main__':
    main()
