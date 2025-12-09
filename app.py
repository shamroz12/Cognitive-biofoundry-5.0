
import streamlit as st
import numpy as np, os, torch
import matplotlib.pyplot as plt
from utils import load_residual, load_ppo_policy, step_mech

st.set_page_config(layout='wide', page_title='Cognitive Biofoundry PPO Demo')
st.title('Cognitive Biofoundry — Hybrid Twin + PPO Demo')

# Sidebar
st.sidebar.header('Simulation Parameters')
mu_max = st.sidebar.slider('mu_max', 0.1, 1.0, 0.5)
K = st.sidebar.slider('Ks', 0.01, 2.0, 0.1)
Yxs = st.sidebar.slider('Y_X/S', 0.1, 1.0, 0.5)
Ypx = st.sidebar.slider('Y_P/X', 0.01, 0.5, 0.1)
S_in = st.sidebar.number_input('Substrate in Feed', value=20.0)
dt = st.sidebar.number_input('Time Step', 0.1, 2.0, 0.5)
params = {'mu_max':mu_max, 'K':K, 'Yxs':Yxs, 'Ypx':Ypx, 'S_in':S_in, 'm':0.01}

st.sidebar.header('Controller Settings')
use_ppo = st.sidebar.checkbox('Use PPO Policy (models/ppo_model.zip)', value=os.path.exists('models/ppo_model.zip'))
fixed_feed = st.sidebar.slider('Fixed Feed Rate', 0.0, 0.2, 0.05)

# Initial state
X0 = st.sidebar.number_input('X0', value=0.1)
S0 = st.sidebar.number_input('S0', value=10.0)
P0 = st.sidebar.number_input('P0', value=0.0)
V0 = st.sidebar.number_input('V0', value=1.0)
T = st.sidebar.number_input('Horizon', 10, 500, 100)

residual = load_residual('models/residual_net.pt')
ppo_agent = load_ppo_policy('models/ppo_model.zip')

if st.button('Run Simulation'):
    state = np.array([X0, S0, P0, V0], float)
    traj = [state.copy()]
    actions = []

    for t in range(int(T)):
        if use_ppo and ppo_agent is not None:
            obs = np.array([state[0], state[1], state[2], t/float(T)], dtype=np.float32)
            action, _ = ppo_agent.predict(obs, deterministic=True)
            feed = float(np.clip(action, 0.0, 0.2))
        else:
            feed = fixed_feed

        mech_next = step_mech(state, feed, params, dt)
        if residual is not None:
            x_in = torch.tensor(state[:3], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                res = residual(x_in).squeeze().cpu().numpy()
            mech_next[:3] += res

        state = np.maximum(mech_next, 0.0)
        traj.append(state.copy()); actions.append(feed)

    traj = np.array(traj); t_axis = np.arange(len(traj))*dt

    fig, ax = plt.subplots(1,2, figsize=(12,4))
    ax[0].plot(t_axis, traj[:,0], label='Biomass X', color='#1B4CF5')
    ax[0].plot(t_axis, traj[:,1], label='Substrate S', color='#13C7B9')
    ax[0].plot(t_axis, traj[:,2], label='Product P', color='#0D8F5B')
    ax[0].legend(); ax[0].set_title('State trajectories')
    ax[1].plot(t_axis[:-1], actions, color='#0B6E79'); ax[1].set_title('Feed Profile')
    st.pyplot(fig)
