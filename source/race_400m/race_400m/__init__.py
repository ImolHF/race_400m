# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Python module serving as a project/extension template.
"""

# Register Gym environments.
from .tasks import *

# Register UI extensions.
from .ui_extension_example import *
# source/race_400m/race_400m/__init__.py

import gymnasium as gym

# 导入环境类和配置类
from .envs import RaceEnv
from .tasks.manager_based.race_400m.race_400m_env_cfg import RaceEnvCfg

# ===== 注册环境 =====
gym.register(
    id="Race400m-v0",                    # 环境名称（你给起的）
    entry_point="race_400m.envs:RaceEnv", # 指向环境类
    kwargs={"cfg": RaceEnvCfg()},         # 传入配置
    disable_env_checker=True,             # 禁用检查（避免警告）
)

__all__ = ["RaceEnv", "RaceEnvCfg"]