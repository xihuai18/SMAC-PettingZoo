from smac_pettingzoo.env.smacv2 import SMACv2Env

distribution_config = {
    "n_units": 5,
    "n_enemies": 5,
    "team_gen": {
        "dist_type": "weighted_teams",
        "unit_types": ["marine", "marauder", "medivac"],
        "exception_unit_types": ["medivac"],
        "weights": [0.45, 0.45, 0.1],
        "observe": True,
    },
    "start_positions": {
        "dist_type": "surrounded_and_reflect",
        "p": 0.5,
        "n_enemies": 5,
        "map_x": 32,
        "map_y": 32,
    },
}

env = SMACv2Env(map_name="10gen_terran", capability_config=distribution_config, smacv2_env_args={})
env.reset(42)
