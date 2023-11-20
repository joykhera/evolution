import tensorboard
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
from ray.rllib.utils import check_env
from gym_env.evolution_env import EvolutionEnv
from gymnasium import spaces
import numpy as np
import os

log_dir = os.path.join(os.getcwd(), "training")

# check_env(EvolutionEnv())

env_config = {
    "num_agents": 10,
    # "render_mode": None,
    "render_mode": 'human',
    "map_size": 100,
    "grid_size": 42,
}

episode_length = 200


# Register the custom environment
def env_creator(env_config):
    return EvolutionEnv(
        num_agents=env_config["num_agents"],
        render_mode=env_config["render_mode"],
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
    "num_workers": 0,
    "framework": "tf",
    "timesteps_per_iteration": episode_length,
    # "min_sample_timesteps_per_reporting": episode_length,
    # "min_time_s_per_reporting": 1,
    # "rollout_fragment_length": episode_length,  # Ensure that rollout fragments match your episode length
    # "train_batch_size": episode_length,
    "multiagent": {
        "policies": {
            "policy_0": (
                None,
                spaces.Box(
                    low=0,
                    high=1,
                    shape=(env_config["grid_size"], env_config["grid_size"], 3),
                    dtype=np.float32,
                ),
                spaces.Discrete(5),
                {},
            ),
        },
        "policy_mapping_fn": lambda agent_id, episode, worker, **kwargs: "policy_0",
    },
    "env_config": env_config,
}

env = EvolutionEnv()
# Run the training algorithm
tune.run(
    PPO,
    config=config,
    stop={"training_iteration": 50},
    local_dir=log_dir,
    checkpoint_freq=10,
    checkpoint_at_end=True,
    reuse_actors=True,
)

# Shutdown Ray
ray.shutdown()
