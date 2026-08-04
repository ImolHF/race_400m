# source/race_400m/race_400m/envs/race_env.py

from isaaclab.envs import ManagerBasedRLEnv
from ..tasks.manager_based.race_400m.race_400m_env_cfg import RaceEnvCfg


class RaceEnv(ManagerBasedRLEnv):
    """400米竞速环境"""

    def __init__(self, cfg: RaceEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

    def step(self, action):
        return super().step(action)

    def reset(self, seed=None, options=None):
        return super().reset(seed=seed, options=options)