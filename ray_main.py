import tensorboard
import os
from argparser import parse_args
import numpy as np
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.tune.registry import register_env
# from gym_env.evolution_env import EvolutionEnv
from evolution_env.pz_env import env_builder, EvolutionEnv
from gymnasium import spaces
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.logger import CSVLoggerCallback, JsonLoggerCallback, TBXLoggerCallback
from pettingzoo.test import parallel_api_test

os.environ["RAY_DEDUP_LOGS"] = "0"


def main():
    args, model_config, env_config = parse_args()
    print("args:", args)
    print("model_config:", model_config)
    print("env_config:", env_config)
    log_dir = os.path.join(os.getcwd(), "training")
    env_name = "EvolutionEnv"
    # env = EvolutionEnv
    # register_env("EvolutionEnv", lambda config: env(config))
    register_env(env_name, lambda env_config: ParallelPettingZooEnv(env_builder(env_config)))
    # register_env(env_name, lambda env_config: ParallelPettingZooEnv(EvolutionEnv(**env_config)))
    print("Environment registered.")
    ray.init()

    # Define the PPOConfig using AlgorithmConfig API.
    config = (
        PPOConfig()
        .environment(env=env_name, env_config=env_config, clip_actions=True)
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
                    spaces.Box(low=0, high=1, shape=(env_config["grid_size"], env_config["grid_size"], 3), dtype=np.float32),
                    spaces.Discrete(5),
                    {},
                )
            },
            policy_mapping_fn=lambda agent_id, episode, **kwargs: "policy_0",
        )
        .resources(num_gpus=0, num_cpus_per_worker=1)
    )

    if args["train"]:
        # parallel_api_test(env, num_cycles=1_000_000)
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

    elif args["test"]:
        print("Testing mode enabled.")
        env_config["render_mode"] = "human"
        # env_config["human_player"] = True
        config.num_env_runners = 0
        trainer = Algorithm.from_checkpoint(args["checkpoint"])
        print("Restored checkpoint from", args["checkpoint"])
        test_env = env_builder(env_config)
        num_episodes = 10
        for episode in range(num_episodes):
            episode_reward = 0
            done = False
            obs, _ = test_env.reset()  # This should return a dictionary of observations
            while not done:
                action_dict = {agent_id: trainer.compute_single_action(agent_obs, policy_id="policy_0") for agent_id, agent_obs in obs.items()}
                obs, rewards, dones, _, _ = test_env.step(action_dict)
                episode_reward += sum(rewards.values())
                done = all(dones.values())
            print(f"Episode {episode} reward: {episode_reward}")

    ray.shutdown()


if __name__ == "__main__":
    main()
