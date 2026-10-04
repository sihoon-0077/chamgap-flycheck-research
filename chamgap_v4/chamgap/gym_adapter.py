"""Optional Gymnasium adapter. Install requirements-optional.txt first.
The numerical core does not require Gymnasium to generate supervised data.
"""
import gymnasium as gym
import numpy as np
from .simulator import SmartFarmCore
from .contracts import D

class SmartFarmEnv(gym.Env):
    metadata = {'render_modes': []}
    def __init__(self):
        super().__init__()
        self.core = SmartFarmCore()
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(-100.,100.,shape=(D,),dtype=np.float32)
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self.core.reset(seed=seed, options=options)
    def step(self, action):
        return self.core.step(int(action))

# The core rejects masked actions. RL trainers must obey action_mask(obs).
# For check_env, use a wrapper translating invalid actions to explicit no-op
# with penalty, or test reset/spaces/reproducibility separately. Do not disable
# physical safety merely to satisfy an unconstrained environment checker.
