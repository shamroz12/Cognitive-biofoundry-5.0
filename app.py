# app.py — Safe Streamlit entry (fallback if torch/SB3 not installed)
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def mu_monod(S, mu_max=0.5, K=0.1):
    return mu_max * S / (K + S + 1e-12)

def mechanistic_step(state, t, feed_rate, params):
    X, S, P, V = state
    mu = mu_monod(S, params['mu_max'], params['K'])
    dX = (mu - params.get('m', 0.0)) * X
    dV = feed_rate
    dS = - (1.0 / params['Yxs']) * mu * X + (feed_rate / max(V,1e-6))*(params.get('S_in',20.0)-S)
    dP = params['Ypx'] * mu * X
    return [dX, dS, dP, dV]

def step_mech(state, feed_rate, params, dt=0.5):
    sol = odeint(mechanistic_step, state, [0,dt], args=(feed_rate,params))
    return np.maximum(sol[-1], 0.0)

st.set_page_config(layout='wide', page_title='Cognitive Biofoundry Demo (Safe Mode)')
st.title('Cognitive Biofoundry — Hybrid Twin Demo (Safe Mode)')

st.sidebar.header('Simulation Parameters')
mu_max = st.sidebar.slider('mu_max', 0.1, 1.0, 0.5)
K = st.sidebar.slider('Ks', 0.01, 2.0, 0.1)
Yxs = st.sidebar.slider('Y_X/S', 0.1, 1.0, 0.5)
Ypx = st.sidebar.slider('Y_P/X', 0.01, 0.5, 0.1)
S_in = st.sidebar.number_input('Substrate in Feed', value=20.0)
dt = st.sidebar.number_input('Time Step', 0.1, 2.0, 0.5)
params = {'mu_max':mu_max, 'K':K, 'Yxs':Yxs, 'Ypx':Ypx, 'S_in':S_in, 'm':0.01}

st.sidebar.header('Controller Settings')
fixed_feed = st.sidebar.slider('Fixed feed rate', 0.0, 0.2, 0.05)

X0 = st.sidebar.number_input('X0', value=0.1)
S0 = st.sidebar.number_input('S0', value=10.0)
P0 = st.sidebar.number_input('P0', value=0.0)
V0 = st.sidebar.number_input('V0', value=1.0)
T = st.sidebar.number_input('Horizon (steps)', 10, 400, 100)

if st.button('Run Simulation'):
    state = np.array([X0, S0, P0, V0], float)
    traj = [state.copy()]; actions = []
    for t in range(int(T)):
        feed = fixed_feed
        state = np.maximum(step_mech(state, feed, params, dt), 0.0)
        traj.append(state.copy()); actions.append(feed)

    traj = np.array(traj); t_axis = np.arange(len(traj))*dt
    fig, ax = plt.subplots(1,2,figsize=(12,4))
    ax[0].plot(t_axis, traj[:,0], label='Biomass X'); ax[0].plot(t_axis, traj[:,1], label='Substrate S'); ax[0].plot(t_axis, traj[:,2], label='Product P')
    ax[0].legend(); ax[0].set_title('State trajectories')
    ax[1].plot(t_axis[:-1], actions); ax[1].set_title('Feed Profile')
    st.pyplot(fig)
