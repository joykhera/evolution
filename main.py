import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
from ray.rllib.utils import check_env
from pettingzoo.classic import rock_paper_scissors_v2
from gym_env.evolution_env import EvolutionEnv
from gym import spaces
import numpy as np


check_env(EvolutionEnv())


# Register the custom environment
def env_creator(env_config):
    return EvolutionEnv(
        num_agents=env_config["num_agents"],
        human_player=env_config["human_player"],
        map_size=env_config["map_size"],
        grid_size=env_config["grid_size"],
    )


def policy_mapping_fn(agent_id, **kwargs):
    return "default_policy"


register_env("EvolutionEnv", env_creator)

# Initialize Ray
ray.init()

# Configure the environment and the PPO agent
config = {
    "env": "EvolutionEnv",
    "num_workers": 1,
    "framework": "tf",
    "multiagent": {
        "policies": {
            "policy_0": (
                None,
                spaces.Box(low=0, high=1, shape=(10, 10, 3), dtype=np.float32),
                spaces.Discrete(5),
                {},
            ),
        },
        # "policy_mapping_fn": policy_mapping_fn,
        # "policy_mapping_fn": lambda agent_id: "policy_0",
        "policy_mapping_fn": lambda agent_id, episode, worker, **kwargs: "policy_0",
    },
    "env_config": {
        "num_agents": 2,
        "human_player": False,
        "map_size": 100,
        "grid_size": 10,
    },
}

env = EvolutionEnv()
# Run the training algorithm
tune.run(PPO, config=config, stop={"training_iteration": 10})

# Shutdown Ray
ray.shutdown()
