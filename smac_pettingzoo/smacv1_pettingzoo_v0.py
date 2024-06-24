import pettingzoo
from __future__ import absolute_import, division, print_function

import atexit
import enum
import math
from copy import deepcopy
from operator import attrgetter

import numpy as np
from absl import logging
from gym.spaces import Discrete
from pysc2 import maps, run_configs
from pysc2.lib import protocol
from s2clientprotocol import common_pb2 as sc_common
from s2clientprotocol import debug_pb2 as d_pb
from s2clientprotocol import raw_pb2 as r_pb
from s2clientprotocol import sc2api_pb2 as sc_pb

from smac.env.starcraft2.maps import get_map_params

races = {
    "R": sc_common.Random,
    "P": sc_common.Protoss,
    "T": sc_common.Terran,
    "Z": sc_common.Zerg,
}

difficulties = {
    "1": sc_pb.VeryEasy,
    "2": sc_pb.Easy,
    "3": sc_pb.Medium,
    "4": sc_pb.MediumHard,
    "5": sc_pb.Hard,
    "6": sc_pb.Harder,
    "7": sc_pb.VeryHard,
    "8": sc_pb.CheatVision,
    "9": sc_pb.CheatMoney,
    "A": sc_pb.CheatInsane,
}

actions = {
    "move": 16,  # target: PointOrUnit
    "attack": 23,  # target: PointOrUnit
    "stop": 4,  # target: None
    "heal": 386,  # Unit
}


class Direction(enum.IntEnum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3


class ParallelEnv(pettingzoo.ParallelEnv):
    metadata = {}

    def __init__(
        self,
        map_name: str,
        step_mul: int = 8,
        move_amount: int = 2,
        difficulty: str = "7",
        game_version: str = None,
        add_local_obs: bool = False,
        add_move_state: bool = False,
        add_visible_state: bool = False,
        add_distance_state: bool = False,
        add_xy_state: bool = False,
        add_enemy_action_state: bool = False,
        add_agent_id: bool = False,
        use_state_agent: bool = True,
        use_mustalive: bool = True,
        add_center_xy: bool = True,
        use_stacked_frames: bool = True,
        stacked_frames: int = 1,
        use_obs_instead_of_state: bool = False,
        continuing_episode=False,
        obs_all_health=True,
        obs_own_health=True,
        obs_last_action=True,
        obs_pathing_grid=False,
        obs_terrain_height=False,
        obs_instead_of_state=False,
        obs_timestep_number=False,
        obs_agent_id=True,
        state_pathing_grid=False,
        state_terrain_height=False,
        state_last_action=True,
        state_timestep_number=False,
        state_agent_id=True,
        reward_sparse=False,
        reward_only_positive=True,
        reward_death_value=10,
        reward_win=200,
        reward_defeat=0,
        reward_negative_scale=0.5,
        reward_scale=True,
        reward_scale_rate=20,
        replay_dir="",
        replay_prefix="",
        window_size_x=1920,
        window_size_y=1200,
        heuristic_ai=False,
        heuristic_rest=False,
        debug=False,
    ):
        """Create a SMACv1 environment following pettingzoo parallel env apis."""
        super().__init__()
        # Map arguments
        self.map_name = map_name
        self.add_local_obs = add_local_obs
        self.add_move_state = add_move_state
        self.add_visible_state = add_visible_state
        self.add_distance_state = add_distance_state
        self.add_xy_state = add_xy_state
        self.add_enemy_action_state = add_enemy_action_state
        self.add_agent_id = add_agent_id
        self.use_state_agent = use_state_agent
        self.use_mustalive = use_mustalive
        self.add_center_xy = add_center_xy
        self.use_stacked_frames = use_stacked_frames
        self.stacked_frames = stacked_frames

        # params from local map_params
        map_params = get_map_params(self.map_name)
        self.n_agents = map_params["n_agents"]
        self.n_enemies = map_params["n_enemies"]
        self.episode_limit = map_params["limit"]
        self._move_amount = move_amount
        self._step_mul = step_mul
        self.difficulty = difficulty
__all__["parallel_env"]
