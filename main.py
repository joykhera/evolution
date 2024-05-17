import tensorboard
import os
import argparse
import numpy as np
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.tune.registry import register_env
from ray.rllib.utils import check_env
from gym_env.evolution_env import EvolutionEnv
from gym_env.simple import SimpleEnv
from gymnasium import spaces
from ray.tune.logger import CSVLoggerCallback, JsonLoggerCallback, TBXLoggerCallback

# It's better to encapsulate the code in functions or a main block to avoid global code execution on import.

env_config = {
    "num_agents": 10,
    "render_mode": None,
    # "render_mode": 'human',
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


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-train", action="store_true", help="Enable training mode.")
    parser.add_argument("-test", action="store_true", help="Enable testing mode.")
    parser.add_argument("-cp", type=str, help="Checkpoint from which to load the model.")
    parser.add_argument("-name", type=str, default="my_experiment", help="Custom name for the experiment.")
    return parser.parse_args()


def train_agent(env_name, config_dict, log_dir, args):
    config_dict["num_workers"] = 9
    tune.run(
        "PPO",
        config=config_dict,
        stop={"training_iteration": 100},
        local_dir=log_dir,
        checkpoint_freq=10,
        checkpoint_at_end=True,
        name=args.name,
        restore=args.cp,
        callbacks=[CSVLoggerCallback(), JsonLoggerCallback(), TBXLoggerCallback()],
    )


def test_agent(env, trainer, num_episodes=10):
    for episode in range(num_episodes):
        episode_reward = 0
        done = False
        obs, _ = env.reset()  # This should return a dictionary of observations
        while not done:
            action_dict = {agent_id: trainer.compute_single_action(agent_obs, policy_id="policy_0") for agent_id, agent_obs in obs.items()}
            obs, rewards, dones, dones, _ = env.step(action_dict)
            episode_reward += sum(rewards.values())
            done = all(dones.values())
        print(f"Episode reward: {episode_reward}")


def main():
    args = parse_arguments()
    log_dir = os.path.join(os.getcwd(), "training")
    env = SimpleEnv
    check_env(env(**env_config))
    register_env("EvolutionEnv", lambda config: env(**config))
    print("Environment registered.")
    env_name = "EvolutionEnv"
    ray.init()

    # Define the PPOConfig directly rather than converting to and from a dictionary.
    config = (
        PPOConfig()
        .environment(env=env_name, env_config=env_config)
        .framework("tf")
        .rollouts(rollout_fragment_length="auto")
        .training(model={"custom_model_config": {}, "conv_filters": [[16, [3, 3], 2], [32, [3, 3], 3], [32, [3, 3], 3]]})
        .multi_agent(
            policies={
                "policy_0": (
                    None,
                    spaces.Box(low=0, high=1, shape=(env_config["grid_size"], env_config["grid_size"], 3), dtype=np.float32),
                    spaces.Discrete(5),
                    {},
                )
            },
            policy_mapping_fn=lambda agent_id, episode, **kwargs: "policy_0",
        )
        .resources(num_gpus=0, num_cpus_per_worker=1)
        .to_dict()
    )  # Only convert to dict when needed for Ray Tune.

    if args.train:
        train_agent(env_name, config, log_dir, args)

    elif args.test:
        if not args.cp:
            raise ValueError("Please specify a checkpoint to load the model from with -cp")
        print("Testing mode enabled.")
        env_config["render_mode"] = "human"
        config["num_workers"] = 0
        trainer = PPO(env=env_name, config=config)
        trainer.restore(args.cp)
        print("Restored checkpoint from", args.cp)
        test_env = env(**env_config)
        test_agent(test_env, trainer)

    ray.shutdown()


if __name__ == "__main__":
    main()
