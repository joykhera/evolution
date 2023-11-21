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
import argparse

log_dir = os.path.join(os.getcwd(), "training")

# check_env(EvolutionEnv())

map_size = 100

env_config = {
    "num_agents": 10,
    "render_mode": None,
    # "render_mode": 'human',
    "map_size": map_size,
    "grid_size": 42,
    "map_color": (255, 255, 255),
    "food_color": (0, 255, 0),
    "player_color": (255, 0, 0),
    "out_of_bounds_color": (0, 0, 0),
    "food_count": 50,
    "fps": 120,
    "food_size": map_size / 50,
    "player_size": map_size / 50,
    "player_size_increase": 0.25,
    "player_speed": map_size / 100,
    "episode_steps": 200,
    "scale": 5,
}

episode_length = 200


# Register the custom environment
def env_creator(env_config):
    return EvolutionEnv(
        num_agents=env_config["num_agents"],
        render_mode=env_config["render_mode"],
        map_size=env_config["map_size"],
        grid_size=env_config["grid_size"],
        map_color=env_config["map_color"],
        food_color=env_config["food_color"],
        player_color=env_config["player_color"],
        out_of_bounds_color=env_config["out_of_bounds_color"],
        food_count=env_config["food_count"],
        fps=env_config["fps"],
        food_size=env_config["food_size"],
        player_size=env_config["player_size"],
        player_size_increase=env_config["player_size_increase"],
        player_speed=env_config["player_speed"],
        episode_steps=env_config["episode_steps"],
        scale=env_config["scale"],
    )


register_env("EvolutionEnv", env_creator)

# Initialize Ray
ray.init()

# Configure the environment and the PPO agent
config = {
    "env": "EvolutionEnv",
    "num_workers": 9,
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-test", action="store_true", help="Enable testing mode.")
    parser.add_argument(
        "-checkpoint", type=str, help="Checkpoint from which to load the model."
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    if args.test and args.checkpoint:
        # Set up the environment for testing
        env_config["render_mode"] = "human"
        env_config["episode_steps"] = 1000
        config["num_workers"] = 0
        env = env_creator(env_config)

        # Create a PPO trainer instance
        trainer = PPO(env="EvolutionEnv", config=config)

        # Restore from the checkpoint
        trainer.restore(args.checkpoint)

        # Run the testing loop
        for _ in range(10):
            episode_reward = 0
            done = False
            obs, _ = env.reset()  # This should return a dictionary of observations

            while not done:
                action_dict = {}
                for agent_id, agent_obs in obs.items():
                    # For each agent, compute an action using its specific observation
                    action_dict[agent_id] = trainer.compute_single_action(
                        agent_obs, policy_id="policy_0"
                    )
                # Step the environment with the actions of all agents
                obs, rewards, dones, _, infos = env.step(action_dict)

                # Accumulate rewards and check if all agents are done
                episode_reward += sum(rewards.values())
                done = all(dones.values())

            print(f"Episode reward: {episode_reward}")
    else:
        tune.run(
            PPO,
            config=config,
            stop={"training_iteration": 100},
            local_dir=log_dir,
            checkpoint_freq=10,
            checkpoint_at_end=True,
            reuse_actors=True,
        )

# Shutdown Ray
ray.shutdown()
