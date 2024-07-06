import numpy as np
from co_mas.test.parallel_api import parallel_api_test, sample_action
from loguru import logger

from smac_pettingzoo import smacv1_pettingzoo_v1

env = smacv1_pettingzoo_v1.parallel_env("8m", {})
ep_i = 0
while True and ep_i < 20:
    obs, info = env.reset(seed=42)
    step = 0
    while True:
        obs, _, terminated, truncated, info = env.step(
            {agent: sample_action(agent, obs[agent], info[agent], env.action_space(agent)) for agent in env.agents}
        )
        step += 1
        if len(env.agents) <= 0:
            logger.debug(f"step {step}, terminated: {terminated} truncated: {truncated}")
            break
    ep_i += 1
    if any(truncated.values()):
        break

env.close()

parallel_api_test(env, 400)

# Seed Tests
env1 = smacv1_pettingzoo_v1.parallel_env("8m", {})
obs1_list = []
obs1, info1 = env1.reset(seed=42)
np.random.seed(42)
obs1_list.append(obs1)

while True:
    obs1, _, terminated1, _, info1 = env1.step(
        {agent: sample_action(agent, obs1[agent], info1[agent], env1.action_space(agent)) for agent in env1.agents}
    )
    obs1_list.append(obs1)

    if any(terminated1.values()):
        break

env1.close()

env2 = smacv1_pettingzoo_v1.parallel_env("8m", {})

obs2_list = []
obs2, info2 = env2.reset(seed=42)
np.random.seed(42)
obs2_list.append(obs2)

while True:
    obs2, _, terminated2, _, info2 = env2.step(
        {agent: sample_action(agent, obs2[agent], info2[agent], env2.action_space(agent)) for agent in env2.agents}
    )
    obs2_list.append(obs2)

    if any(terminated2.values()):
        break

env2.close()

for i, (obs1, obs2) in enumerate(zip(obs1_list, obs2_list)):
    assert all(
        (obs1[agent] == obs2[agent]).all() for agent in env1.agents
    ), f"Observations at step {i} differ:\n{obs1}\n{obs2}"

logger.success("Seed Test Passed!")

# Wrapper Tests
from co_mas.wrappers import AutoResetParallelEnvWrapper, OrderForcingParallelEnvWrapper

env = smacv1_pettingzoo_v1.parallel_env(
    "8m", {}, additional_wrappers=[OrderForcingParallelEnvWrapper, AutoResetParallelEnvWrapper]
)

obs, info = env.reset(seed=42)

while True:
    obs, _, terminated, _, info = env.step(
        {agent: sample_action(agent, obs[agent], info[agent], env.action_space(agent)) for agent in env.agents}
    )

    if all(terminated.values()):
        break

obs, _, terminated, _, info = env.step(
    {agent: sample_action(agent, obs[agent], info[agent], env.action_space(agent)) for agent in env.agents}
)

assert terminated != {agent: True for agent in env.agents}

logger.success("Auto Reset Test Passed!")

logger.success("Wrapper Test Passed!")
env.close()
