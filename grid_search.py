import os
import numpy as np
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.tune.schedulers import ASHAScheduler
from gym_env.simple import SimpleEnv
from gymnasium import spaces

env_config = {
    "num_agents": 10,
    "render_mode": None,
    "map_size": 50,
    "grid_size": 24,
    "map_color": (255, 255, 255),
    "food_color": (0, 255, 0),
    "player_color": (255, 0, 0),
    "out_of_bounds_color": (0, 0, 0),
    "food_count": 30,
    "fps": 120,
    "food_size": 1,
    "player_size": 1,
    "player_size_increase": 0.25,
    "player_speed": 0.5,
    "decay_rate": 0.001,
    "episode_length": 200,
    "scale": 10,
}


def register_and_check_env():
    env = SimpleEnv
    register_env("EvolutionEnv", lambda config: env(**config))
    print("Environment registered.")


register_and_check_env()


def trainable(config):
    config_dict = PPOConfig().to_dict()
    config_dict.update(config)
    trainer = PPOConfig().environment("EvolutionEnv", env_config=env_config).build()
    while True:
        result = trainer.train()
        tune.report(mean_reward=result["episode_reward_mean"])


search_space = {
    "num_workers": 9,
    "framework": "tf",
    "rollout_fragment_length": tune.choice([200, "auto"]),
    "train_batch_size": tune.choice([4000, 4440]),  # Adjust to match rollout_fragment_length
    "model": {
        "conv_filters": tune.grid_search(
            [
                [[16, [3, 3], 2], [32, [3, 3], 2]],
                [[32, [5, 5], 1], [64, [5, 5], 2]],
                [[32, [3, 3], 2], [64, [3, 3], 3], [128, [3, 3], 2]],
            ]
        )
    },
    "multi_agent": {
        "policies": {"policy_0": (None, spaces.Box(low=0, high=1, shape=(env_config["grid_size"], env_config["grid_size"], 3), dtype=np.float32), spaces.Discrete(5), {})},
        "policy_mapping_fn": lambda agent_id, **kwargs: "policy_0",
    },
    "resources": {"num_gpus": 0, "num_cpus_per_worker": 1},
}

local_dir = "file://" + os.path.join(os.getcwd(), "ray_results")

analysis = tune.run(trainable, config=search_space, num_samples=10, scheduler=ASHAScheduler(metric="mean_reward", mode="max"), local_dir=local_dir)

print("Best config: ", analysis.get_best_config(metric="mean_reward", mode="max"))
