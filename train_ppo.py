
# train_ppo.py - trains PPO agent on the custom Gymnasium env
import os
from envs import FedBatchEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
from utils import ResidualNet
import numpy as np

OUT = 'models'; os.makedirs(OUT, exist_ok=True)

def generate_residual_data(params, num_rollouts=20, rollout_length=120, dt=0.5):
    Xs, Ys = [], []
    for r in range(num_rollouts):
        def feed_profile(t): return 0.05 * (1.0 + np.sin(0.03*(t + r*5)))
        fps = [feed_profile(i*dt) for i in range(rollout_length)]
        state = [0.1, 10.0, 0.0, 1.0]
        traj = []
        for fr in fps:
            from scipy.integrate import odeint
            sol = odeint(lambda s, tt: mechanistic_step(s, tt, fr, params), state, [0, dt])
            state = sol[-1]; traj.append(state.copy())
        for i in range(len(traj)-1):
            Xs.append(traj[i][:3]); Ys.append(np.array(traj[i+1][:3]) - np.array(traj[i][:3]))
    return np.array(Xs, dtype=float), np.array(Ys, dtype=float)

if __name__ == '__main__':
    params = {'mu_max':0.5,'K':0.1,'Yxs':0.5,'Ypx':0.1}
    # train a small residual net (optional) - here we skip or load if exists
    res = ResidualNet()
    torch.save(res.state_dict(), os.path.join(OUT, 'residual_net.pt'))
    print('Saved placeholder residual net to models/residual_net.pt')

    # create env and train PPO
    env = DummyVecEnv([lambda: FedBatchEnv(params=params, dt=0.5, episode_length=100)])
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=50000)
    model.save(os.path.join(OUT, 'ppo_model'))
    print('PPO training complete. Saved to models/ppo_model.zip')
