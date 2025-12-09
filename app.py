# app.py — Safe Streamlit entry (graceful fallback if torch / SB3 not installed)
import streamlit as st
import numpy as np, os
import matplotlib.pyplot as plt

# Try importing optional heavy dependencies
torch = None
ppo_agent = None
residual = None
sb3_available = False

try:
    import torch as _torch
    torch = _torch
except Exception as e:
    torch = None

# We will try (optionally) to load PPO if stable-baselines3 is present.
def try_load_ppo(path):
    try:
        from stable_baselines3 import PPO
        model = PPO.load(path, device="cpu")
        return model
    except Exception:
        return None

# Small mechanistic functions (self-contained)
from scipy.integrate import odeint

def mu_monod(S, mu_max=0.5, K=0.1):
    return mu_max * S / (K + S + 1e-12)

def mechanistic_step(state, t, feed_rate, params):
    X, S, P, V = state
    mu = mu_monod(S, params['mu_max'], params['K'])
    dX = (mu - params.get('m', 0.0)) * X
    dV = feed_rate
    dS = - (1.0 / params['Yxs']) * mu * X + (feed_rate / max(V, 1e-6)) * (params.get('S_in', 20.0) - S)
    dP = params['Ypx'] * mu * X
    return [dX, dS, dP, dV]

def step_mech(state, feed_rate, params, dt=0.5):
    sol = odeint(mechanistic_step, state, [0, dt], args=(feed_rate, params))
    return np.maximum(sol[-1], 0.0)

# Try to load optional models only if torch is available
if torch is not None:
    try:
        from utils import load_residual, load_ppo_policy  # your utils loader
    except Exception:
        # fallback: try local simple loader implementations (if your utils are missing)
        load_residual = lambda p: None
        load_ppo_policy = lambda p: try_load_ppo(p)
    try:
        residual = load_residual("models/residual_net.pt")
    except Exception:
        residual = None
    try:
        # prefer PPO artifact name used in repo
        ppo_agent = load_ppo_policy("models/ppo_model.zip") or load_ppo_policy("models/ppo_model")
        if ppo_agent is None:
            # try zipped SB3 model name
            ppo_agent = load_ppo_policy("models/ppo_model.zip")
    except Exception:
        ppo_agent = None
else:
    residual = None
    ppo_agent = None

# Streamlit UI
st.set_page_config(layout='wide', page_title='Cognitive Biofoundry Demo (Safe Mode)')
st.title('Cognitive Biofoundry — Hybrid Twin Demo')

# Notice to user about missing packages
if torch is None:
    st.warning(
        "Optional ML libraries (torch / stable-baselines3) are not installed in this environment. "
        "The app is running in reduced mode using the mechanistic model only. "
        "For full functionality (PPO policy + residual net), run locally or deploy with Docker (instructions in README)."
    )

# Sidebar controls
st.sidebar.header('Simulation Parameters')
mu_max = st.sidebar.slider('mu_max', 0.1, 1.0, 0.5)
K = st.sidebar.slider('Ks', 0.01, 2.0, 0.1)
Yxs = st.sidebar.slider('Y_X/S', 0.1, 1.0, 0.5)
Ypx = st.sidebar.slider('Y_P/X', 0.01, 0.5, 0.1)
S_in = st.sidebar.number_input('Substrate in Feed', value=20.0)
dt = st.sidebar.number_input('Time Step', 0.1, 2.0, 0.5)
params = {'mu_max': mu_max, 'K': K, 'Yxs': Yxs, 'Ypx': Ypx, 'S_in': S_in, 'm': 0.01}

st.sidebar.header('Controller Settings')
use_ppo = st.sidebar.checkbox('Use PPO (if available)', value=(ppo_agent is not None))
fixed_feed = st.sidebar.slider('Fixed feed rate (if RL disabled)', 0.0, 0.2, 0.05)

# Initial state
X0 = st.sidebar.number_input('Initial Biomass X0', value=0.1)
S0 = st.sidebar.number_input('Initial Substrate S0', value=10.0)
P0 = st.sidebar.number_input('Initial Product P0', value=0.0)
V0 = st.sidebar.number_input('Initial Volume V0', value=1.0)
T = st.sidebar.number_input('Horizon (steps)', 10, 400, 100)

# Run simulation
if st.button('Run Simulation'):
    state = np.array([X0, S0, P0, V0], float)
    traj = [state.copy()]
    actions = []
    for t in range(int(T)):
        # choose control
        if use_ppo and ppo_agent is not None:
            # SB3 expects batched inputs sometimes; convert if needed
            try:
                obs = np.array([state[0], state[1], state[2], t / float(T)], dtype=np.float32)
                action, _ = ppo_agent.predict(obs, deterministic=True)
                feed = float(np.clip(action, 0.0, 0.2))
            except Exception:
                feed = fixed_feed
        else:
            feed = fixed_feed

        mech_next = step_mech(state, feed, params, dt)
        # apply residual correction only if available
        if residual is not None and torch is not None:
            try:
                x_in = torch.tensor(state[:3], dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    res = residual(x_in).squeeze().cpu().numpy()
                mech_next[:3] += res
            except Exception:
                pass

        state = np.maximum(mech_next, 0.0)
        traj.append(state.copy())
        actions.append(feed)

    traj = np.array(traj); t_axis = np.arange(len(traj)) * dt

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(t_axis, traj[:, 0], label='Biomass X', color='#1B4CF5')
    ax[0].plot(t_axis, traj[:, 1], label='Substrate S', color='#13C7B9')
    ax[0].plot(t_axis, traj[:, 2], label='Product P', color='#0D8F5B')
    ax[0].legend(); ax[0].set_title('State trajectories')
    ax[1].plot(t_axis[:-1], actions, color='#0B6E79'); ax[1].set_title('Feed Profile')
    st.pyplot(fig)
