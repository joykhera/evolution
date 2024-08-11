import os
import ray
import time
import wandb
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


def train_ppo(env_config):
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
        .rollouts(rollout_fragment_length="auto")
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
        config=config.to_dict(),
        stop={"training_iteration": 200},
        storage_path=os.path.join(os.getcwd(), "training"),
        checkpoint_at_end=True,
        checkpoint_freq=0,
        callbacks=[WandbLoggerCallback(project="custom_tag", log_config=True)],
        verbose=0,
    )

    ray.shutdown()
    return results


def test_ppo(env_config, checkpoint_path):
    ray.init(ignore_reinit_error=True, log_to_driver=False)
    trainer = Algorithm.from_checkpoint(checkpoint_path)

    for episode in range(10):
        env_config["render_mode"] = "human"
        env = TagEnv(**env_config)
        episode_predator_reward = 0
        episode_prey_reward = 0
        done = False
        observations, _ = env.reset()  # This should return a dictionary of observations
        while not done:
            actions = {agent: trainer.compute_single_action(observations[agent], policy_id=get_policy_mapping_fn(agent, None)) for agent in env.agents}
            observations, rewards, terminations, truncations, infos = env.step(actions)
            # print(observations, actions, rewards)
            episode_predator_reward += sum(rewards[predator] for predator in rewards if "predator" in predator) / env_config["num_predators"]
            episode_prey_reward += sum(rewards[prey] for prey in rewards if "prey" in prey) / env_config["num_prey"]
            env.render()
            time.sleep(0.05)
            done = any(terminations.values()) or any(truncations.values())
        print(f"Episode {episode}, predator reward: {episode_predator_reward}, prey reward: {episode_prey_reward}")

    ray.shutdown()


if __name__ == "__main__":
    run_name = "tag_1each_newobs"
    env_config = {
        "num_prey": 1,  # Only one prey
        "num_predators": 1,  # Three predators
        "grid_size": 50,
        "max_steps": 500,
        "screen_size": 600,
    }
    # register_env("custom_tag_v0", lambda config: env_creator(config))
    # results = train_ppo(env_config)
    # checkpoint = results.get_best_checkpoint(f"training/{run_name}", metric="episode_reward_mean", mode="max")
    checkpoint = "training/tag_1each_newobs/PPO_custom_tag_v0_abfb9_00000_0_2024-08-11_11-30-43/checkpoint_000000"
    test_ppo(env_config, checkpoint)
