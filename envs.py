
import numpy as np
from gymnasium import spaces, Env
from scipy.integrate import odeint
from utils import mechanistic_step, step_mech

class FedBatchEnv(Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, params=None, dt=0.5, episode_length=100):
        super().__init__()
        self.params = params or {'mu_max':0.5,'K':0.1,'Yxs':0.5,'Ypx':0.1}
        self.dt = dt
        self.episode_length = episode_length
        self.max_feed = 0.2
        # observation: X, S, P, t_norm
        high = np.array([1e3, 1e3, 1e3, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=high, dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=self.max_feed, shape=(1,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.state = np.array([0.1, 10.0, 0.0, 1.0], dtype=float)
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([self.state[0], self.state[1], self.state[2], self.t/self.episode_length], dtype=np.float32)

    def step(self, action):
        feed_rate = float(action[0]) if isinstance(action, (list, np.ndarray)) else float(action)
        mech_next = step_mech(self.state, feed_rate, self.params, self.dt)
        self.state = np.maximum(mech_next, 0.0)
        deltaP = self.state[2] - 0.0  # reward uses product increment over step (approx)
        reward = float(deltaP - 0.1*feed_rate)
        self.t += 1
        done = (self.t >= self.episode_length)
        return self._get_obs(), reward, done, False, {}
