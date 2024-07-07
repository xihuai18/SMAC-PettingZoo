from collections import defaultdict
from typing import Any, Dict, List, Tuple, Type

import co_mas
import gymnasium as gym
import numpy as np
import pettingzoo
import pettingzoo.utils
from loguru import logger

from smac_pettingzoo.env.smacv1 import SMACv1Env


class ParallelEnv(co_mas.env.ParallelEnv):
    """
    Parallel Apis of SMACv1
    """

    metadata = {}

    def __init__(self, map_name: str, smacv1_env_args: dict = {}):
        self._env = SMACv1Env(map_name, **smacv1_env_args)
        # NOTE: To obtain agent infos, should be reset again.
        self._env.reset(0)
        self._init_agents()

        self.observation_spaces = gym.spaces.Dict(
            {
                agent: gym.spaces.Box(
                    low=-1, high=1, shape=(self._env.observation_space[agent_i][0],), dtype=np.float32
                )
                for agent_i, agent in enumerate(self.agents)
            }
        )
        self.action_spaces = gym.spaces.Dict(
            {agent: self._env.action_space[agent_i] for agent_i, agent in enumerate(self.agents)}
        )
        self.state_spaces = gym.spaces.Dict(
            {
                agent: gym.spaces.Box(
                    low=-1, high=1, shape=(self._env.share_observation_space[agent_i][0],), dtype=np.float32
                )
                for agent_i, agent in enumerate(self.agents)
            }
        )

        self.states = None

    def _init_agents(self):
        self.agents: List[str] = []

        agent_type_count: Dict[str, int] = defaultdict(lambda: 0)

        for agent_id in range(self._env.n_agents):
            agent_info = self._env.agents[agent_id]
            match agent_info.unit_type:
                case self._env.marine_id:
                    agent_type = "marine"
                case self._env.marauder_id:
                    agent_type = "marauder"
                case self._env.medivac_id:
                    agent_type = "medivac"
                case self._env.hydralisk_id:
                    agent_type = "hydralisk"
                case self._env.zergling_id:
                    agent_type = "zergling"
                case self._env.baneling_id:
                    agent_type = "baneling"
                case self._env.stalker_id:
                    agent_type = "stalker"
                case self._env.colossus_id:
                    agent_type = "colossus"
                case self._env.zealot_id:
                    agent_type = "zealot"
                case _:
                    raise NotImplementedError(f"Unknown unit type: {agent_info.unit_type}")

            self.agents.append(f"{agent_type}_{agent_type_count[agent_type]}")
            agent_type_count[agent_type] += 1

        logger.trace(f"Agents in SMACv1: {self.agents}")

        self.possible_agents = self.agents[:]
        self.agents_to_agent_ids = {agent: agent_id for agent_id, agent in enumerate(self.possible_agents)}

    def observation_space(self, agent: Any) -> gym.spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: Any) -> gym.spaces.Space:
        return self.action_spaces[agent]

    def state_space(self, agent: Any) -> gym.spaces.Space:
        return self.state_spaces[agent]

    def state(self) -> Dict:

        assert self.states is not None, "Run `reset` or `step` first"

        return self.states

    def reset(self, seed: int = None, options: Dict = None) -> Tuple[Dict]:
        super().reset(seed=seed, options=options)
        self.agents = self.possible_agents[:]
        _observations, _states, _action_masks = self._env.reset(seed=self.np_random.integers(0, 2000000000, dtype=int))

        self.states = {agent: state for agent, state in zip(self.agents, _states)}
        observations = {agent: obs for agent, obs in zip(self.agents, _observations)}
        infos = {
            agent: {
                "action_mask": action_mask,
            }
            for agent, action_mask in zip(self.agents, _action_masks)
        }

        return observations, infos

    def step(self, actions: Dict) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        # no-op, stop, move * 4, attack * n_enemies or heal * n_allies
        _actions = [0 for _ in range(self.max_num_agents)]
        for agent, action in actions.items():
            _actions[self.agents_to_agent_ids[agent]] = action
        _observations, _states, _rewards, _terminations, _infos, _action_masks = self._env.step(_actions)
        observations = {agent: obs for agent, obs in zip(self.possible_agents, _observations) if agent in self.agents}
        self.states = {agent: state for agent, state in zip(self.possible_agents, _states) if agent in self.agents}
        rewards = {agent: reward for agent, reward in zip(self.possible_agents, _rewards) if agent in self.agents}
        terminations = {
            agent: termination
            for agent, termination in zip(self.possible_agents, _terminations)
            if agent in self.agents
        }
        truncations = {}
        for agent, _info in zip(self.possible_agents, _infos):
            if agent in self.agents:
                truncations[agent] = _info.pop("truncation", False)
        infos = {
            agent: {
                "action_mask": action_mask,
                **info,
            }
            for agent, action_mask, info in zip(self.possible_agents, _action_masks, _infos)
            if agent in self.agents
        }

        self.agents = [agent for agent in self.agents if (not terminations[agent] and not truncations[agent])]

        return observations, rewards, terminations, truncations, infos

    def close(self):
        if hasattr(self, "_env") and self._env is not None:
            self._env.close()


def parallel_env(
    map_name: str,
    smacv1_env_args: dict = {},
    order_forcing: bool = True,
    additional_wrappers: List[Type[pettingzoo.utils.BaseParallelWrapper]] = [],
) -> ParallelEnv:
    env = ParallelEnv(map_name, smacv1_env_args)

    from co_mas.wrappers import OrderForcingParallelEnvWrapper

    if order_forcing and not any(
        issubclass(wrapper, OrderForcingParallelEnvWrapper) for wrapper in additional_wrappers
    ):
        env = OrderForcingParallelEnvWrapper(env)

    for wrapper in additional_wrappers:
        if issubclass(wrapper, pettingzoo.utils.BaseParallelWrapper):
            env = wrapper(env)
        elif issubclass(wrapper, pettingzoo.utils.BaseWrapper):
            from co_mas.wrappers import AECToParallelWrapper, ParallelToAECWrapper

            aec_env = ParallelToAECWrapper(env)
            aec_env = wrapper(aec_env)
            env = AECToParallelWrapper(aec_env)
        else:
            raise ValueError(f"Unknown wrapper type: {wrapper}")

    return env


__all__ = ["parallel_env"]
