import os
import ray
import time
import argparse
from tag_env import TagEnv
from ray import tune
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune.registry import register_env
from ray.rllib.env import ParallelPettingZooEnv
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig


# Register the environment
def env_creator(config):
    return ParallelPettingZooEnv(TagEnv(**config))


register_env("custom_tag_v0", env_creator)


def get_policy_mapping_fn(agent_id, episode, **kwargs):
    if "predator" in agent_id:
        return "predator_policy"
    else:
        return "prey_policy"


def train_ppo(env_config, run_name):
    ray.init(ignore_reinit_error=True, log_to_driver=False)

    # Create the environment to fetch the observation and action spaces
    env = TagEnv(**env_config)
    observation_spaces = env.observation_spaces
    action_spaces = env.action_spaces

    # Prepare the multi-agent policies dynamically based on the number of prey/predators
    policies = {
        "prey_policy": (None, observation_spaces["prey_0"], action_spaces["prey_0"], {}),
        "predator_policy": (None, observation_spaces["predator_0"], action_spaces["predator_0"], {}),
    }

    config = (
        PPOConfig()
        .environment(env="custom_tag_v0", env_config=env_config, clip_actions=True)
        .framework("torch")
        .env_runners(rollout_fragment_length="auto")
        .training(
            # lr=tune.grid_search([0.01, 0.001, 0.0001]),
            lr=0.00001
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=get_policy_mapping_fn,
        )
    )
    config.num_env_runners = 9
    print("Starting training for run:", run_name)
    results = tune.run(
        "PPO",
        name=run_name,
        # metric="episode_reward_mean",
        # mode="max",
        config=config.to_dict(),
        stop={"training_iteration": 100},
        storage_path=os.path.join(os.getcwd(), "training"),
        checkpoint_at_end=True,
        checkpoint_freq=0,
        callbacks=[WandbLoggerCallback(project="custom_tag", log_config=True, name=run_name)],
        # verbose=0,
        reuse_actors=True,
    )

    ray.shutdown()
    return results


def test_ppo(env_config, checkpoint_path):
    ray.init(ignore_reinit_error=True, log_to_driver=False)
    trainer = Algorithm.from_checkpoint(checkpoint_path)
    env_config["render_mode"] = "human"
    env = TagEnv(**env_config)

    for episode in range(10):
        episode_predator_reward = 0
        episode_prey_reward = 0
        done = False
        observations, _ = env.reset()  # This should return a dictionary of observations
        while not done:
            actions = {agent: trainer.compute_single_action(observations[agent], policy_id=get_policy_mapping_fn(agent, None)) for agent in env.agents}
            observations, rewards, terminations, truncations, infos = env.step(actions)
            # print('sss', observations)
            print('sss', rewards)
            episode_prey_reward += sum(rewards[prey] for prey in rewards if "prey" in prey) / env_config["num_prey"]
            episode_predator_reward += sum(rewards[predator] for predator in rewards if "predator" in predator) / env_config["num_predators"]
            time.sleep(0.05)
            done = any(terminations.values()) or any(truncations.values())
        print(f"Episode {episode}, predator reward: {episode_predator_reward}, prey reward: {episode_prey_reward}")

    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-train", action="store_true", help="Flag to train the model")
    parser.add_argument("-test", action="store_true", help="Flag to test the model")
    parser.add_argument("-cp", "--checkpoint", type=str, help="Checkpoint path for testing")
    parser.add_argument("-rn", "--run_name", type=str, help="Run name")
    args = parser.parse_args()

    env_config = {
        "num_prey": 2,
        "num_predators": 2,
        "prey_speed": 1,
        "predator_speed": 1,
        "map_size": 30,
        "max_steps": 200,
        "screen_size": 600,
        "prey_view_size": 10,
        "predator_view_size": 10,
    }

    if args.train:
        run_name = args.run_name or "no_name"
        results = train_ppo(env_config, run_name)
        # checkpoint = results.get_best_checkpoint(f"training/{run_name}", metric="episode_reward_mean", mode="max")
        # checkpoint = results.best_checkpoint
        # print(f"Training complete. Best checkpoint: {checkpoint}")
    elif args.test:
        if args.checkpoint is None:
            print("Checkpoint path must be provided for testing mode.")
        else:
            test_ppo(env_config, args.checkpoint)
    else:
        print("Please provide either --train or --test flag.")
