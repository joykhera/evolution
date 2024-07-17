import tensorboard
import os
from argparser import parse_args
import numpy as np
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.tune.registry import register_env
from gym_env.evolution_env import EvolutionEnv
from gymnasium import spaces
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.logger import CSVLoggerCallback, JsonLoggerCallback, TBXLoggerCallback


def train_agent(config, log_dir, args, model_config):
    config.num_env_runners = model_config["num_runners"]
    tune.run(
        "PPO",
        config=config.to_dict(),
        stop={"training_iteration": model_config["training_iterations"]},
        storage_path=log_dir,
        checkpoint_freq=10,
        checkpoint_at_end=True,
        name=args["save_name"],
        restore=args["checkpoint"],
        reuse_actors=True,
        log_to_file=True,
        # callbacks=[CSVLoggerCallback(), JsonLoggerCallback(), TBXLoggerCallback()],
    )


def test_agent(env, trainer, num_episodes=10):
    for episode in range(num_episodes):
        episode_reward = 0
        done = False
        obs, _ = env.reset()  # This should return a dictionary of observations
        while not done:
            action_dict = {agent_id: trainer.compute_single_action(agent_obs, policy_id="policy_0") for agent_id, agent_obs in obs.items()}
            obs, rewards, dones, _, _ = env.step(action_dict)
            episode_reward += sum(rewards.values())
            done = all(dones.values())
        print(f"Episode {episode} reward: {episode_reward}")


def main():
    args, model_config, env_config = parse_args()
    print(args, model_config, env_config)
    log_dir = os.path.join(os.getcwd(), "training")
    env = EvolutionEnv
    register_env("EvolutionEnv", lambda config: env(**config))
    print("Environment registered.")
    env_name = "EvolutionEnv"
    ray.init()

    # Define the PPOConfig using AlgorithmConfig API.
    config = (
        PPOConfig()
        .environment(env=env_name, env_config=env_config)
        .framework("torch")
        .env_runners(rollout_fragment_length="auto")
        .training(
            model={
                "custom_model_config": {},
                "conv_filters": [[16, [3, 3], 2], [32, [3, 3], 3], [32, [3, 3], 3]],
            }
        )
        .multi_agent(
            policies={
                "policy_0": (
                    None,
                    # spaces.Box(low=0, high=1, shape=(env_config["grid_size"], env_config["grid_size"], 3), dtype=np.float32),
                    spaces.Dict(
                        {
                            "visual": spaces.Box(low=0, high=1, shape=(env_config["grid_size"], env_config["grid_size"], 3), dtype=np.float32),
                            "boost_info": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                        }
                    ),
                    spaces.Tuple(
                        [
                            spaces.Discrete(5),  # Direction to move in
                            spaces.Discrete(2),  # Boost or not (0 or 1)
                        ]
                    ),
                    {},
                )
            },
            policy_mapping_fn=lambda agent_id, episode, **kwargs: "policy_0",
        )
        .resources(num_gpus=0, num_cpus_per_worker=1)
    )

    if args["train"]:
        train_agent(config, log_dir, args, model_config)

    elif args["test"]:
        print("Testing mode enabled.")
        env_config["render_mode"] = "human"
        # env_config["human_player"] = True
        config.num_env_runners = 0
        trainer = Algorithm.from_checkpoint(args["checkpoint"])
        print("Restored checkpoint from", args["checkpoint"])
        test_env = env(**env_config)
        test_agent(test_env, trainer)

    ray.shutdown()


if __name__ == "__main__":
    main()
