
import numpy as np, torch, os
from scipy.integrate import odeint
import torch.nn as nn

def mu_monod(S, mu_max=0.5, K=0.1):
    return mu_max * S / (K + S + 1e-12)

def mechanistic_step(state, t, feed_rate, params):
    X,S,P,V = state
    mu = mu_monod(S, params['mu_max'], params['K'])
    dX = (mu - params.get('m',0.0)) * X
    dV = feed_rate
    dS = - (1.0 / params['Yxs']) * mu * X + (feed_rate/max(V,1e-6))*(params.get('S_in',20.0)-S)
    dP = params['Ypx'] * mu * X
    return [dX, dS, dP, dV]

def step_mech(state, feed_rate, params, dt=0.5):
    sol = odeint(mechanistic_step, state, [0,dt], args=(feed_rate,params))
    return np.maximum(sol[-1], 0.0)

class ResidualNet(nn.Module):
    def __init__(self, in_dim=3, hidden=128, out_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x): return self.net(x)

def load_residual(path):
    if not os.path.exists(path): return None
    model = ResidualNet(in_dim=3, hidden=128, out_dim=3)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval(); return model

def load_ppo_policy(path):
    try:
        from stable_baselines3 import PPO
    except Exception as e:
        print('stable_baselines3 not installed:', e); return None
    if not os.path.exists(path): return None
    model = PPO.load(path, device='cpu'); return model
