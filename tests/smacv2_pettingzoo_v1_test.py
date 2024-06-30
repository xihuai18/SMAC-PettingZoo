from co_mas.test import parallel_api_test, sample_action
from loguru import logger

from smac_pettingzoo import smacv2_pettingzoo_v1

env = smacv2_pettingzoo_v1.parallel_env("10gen_terran_10_vs_10")

while True:
    obs, info = env.reset(seed=42)
    step = 0
    while True:
        obs, _, terminated, truncated, info = env.step(
            {agent: sample_action(env, obs, agent, info) for agent in env.agents}
        )
        step += 1
        if len(env.agents) <= 0:
            logger.debug(f"step {step}, terminated: {terminated} truncated: {truncated}")
            break
    if any(truncated.values()):
        break
env.close()

parallel_api_test(env, 200)

# Seed Tests
env1 = smacv2_pettingzoo_v1.parallel_env("10gen_terran_10_vs_10")
obs1_list = []
obs1, info1 = env1.reset(seed=42)
obs1_list.append(obs1)

while True:
    obs1, _, terminated1, truncated1, info1 = env1.step(
        {agent: sample_action(env1, obs1, agent, info1) for agent in env1.agents}
    )
    obs1_list.append(obs1)
    logger.trace(f"{env1.agents}")

    if len(env1.agents) <= 0:
        break

env1.close()

env2 = smacv2_pettingzoo_v1.parallel_env("10gen_terran_10_vs_10")

obs2_list = []
obs2, info2 = env2.reset(seed=42)
obs2_list.append(obs2)

while True:
    obs2, _, terminated2, _, info2 = env2.step(
        {agent: sample_action(env2, obs2, agent, info2) for agent in env2.agents}
    )
    obs2_list.append(obs2)

    if len(env2.agents) <= 0:
        break

for i, (obs1, obs2) in enumerate(zip(obs1_list, obs2_list)):
    assert all(
        (obs1[agent] == obs2[agent]).all() for agent in env1.agents
    ), f"Observations at step {i} differ:\n{obs1}\n{obs2}"

logger.success("Seed Test Passed!")
env1.close()
env2.close()

# Wrapper Tests
from co_mas.wrappers import AutoResetParallelEnvWrapper, OrderForcingParallelEnvWrapper

env = smacv2_pettingzoo_v1.parallel_env(
    "10gen_terran_10_vs_10", additional_wrappers=[OrderForcingParallelEnvWrapper, AutoResetParallelEnvWrapper]
)

obs, info = env.reset(seed=42)

while True:
    obs, _, terminated, _, info = env.step({agent: sample_action(env, obs, agent, info) for agent in env.agents})

    if len(env.agents) <= 0:
        break

obs, _, terminated, _, info = env.step({agent: sample_action(env, obs, agent, info) for agent in env.agents})

assert terminated != {agent: True for agent in env.agents}

logger.success("Auto Reset Test Passed!")

logger.success("Wrapper Test Passed!")
env.close()