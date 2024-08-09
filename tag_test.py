import os
import ray
import time
from tag_env import TagEnv
from ray import tune
from ray.tune.registry import register_env
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.rllib.env import ParallelPettingZooEnv
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
import random


# Register the environment
def env_creator(config):
    return ParallelPettingZooEnv(TagEnv())


register_env("custom_tag_v0", env_creator)

def get_policy_mapping_fn(agent_id, episode, **kwargs):
    if "adversary" in agent_id:
        return "adversary_policy"
    else:
        return "agent_policy"


def train_ppo(observation_spaces, action_spaces):
    ray.init(ignore_reinit_error=True, log_to_driver=False)
    config = (
        PPOConfig()
        .environment(env="custom_tag_v0", clip_actions=True)
        .framework("torch")
        .rollouts(rollout_fragment_length="auto")
        .multi_agent(
            policies={
                "adversary_policy": (None, observation_spaces["adversary_0"], action_spaces["adversary_0"], {}),
                "agent_policy": (None, observation_spaces["agent_0"], action_spaces["agent_0"], {}),
            },
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
        callbacks=[
            MLflowLoggerCallback(
                experiment_name=run_name,
                save_artifact=True,
            )
        ],
        # verbose=0,
    )

    ray.shutdown()
    return results


def test_ppo(env, checkpoint_path):
    ray.init(ignore_reinit_error=True, log_to_driver=False)
    trainer = Algorithm.from_checkpoint(checkpoint_path)

    for episode in range(10):
        episode_adversary_reward = 0
        episode_agent_reward = 0
        done = False
        observations, _ = env.reset()  # This should return a dictionary of observations
        while not done:
            actions = {agent: trainer.compute_single_action(observations[agent], policy_id=get_policy_mapping_fn(agent, None)) for agent in env.agents}
            observations, rewards, terminations, truncations, infos = env.step(actions)
            # print(actions, rewards)
            episode_adversary_reward += (rewards["adversary_0"] + rewards["adversary_1"]) / 2
            episode_agent_reward += rewards["agent_0"]
            env.render()
            time.sleep(0.05)
            done = any(terminations.values()) or any(truncations.values())
        print(f"Episode {episode}, adversary reward: {episode_adversary_reward}, agent reward: {episode_agent_reward}")

    ray.shutdown()


if __name__ == "__main__":
    run_name = "tagtest"
    env = TagEnv()
    observation_spaces = env.observation_spaces
    action_spaces = env.action_spaces
    # results = train_ppo(observation_spaces, action_spaces)
    # checkpoint = results.get_best_result(metric="episode_reward_mean", mode="max").checkpoint
    # checkpoint = results.get_best_checkpoint(results.get_last_trial(), mode="max")
    checkpoint = "training/tagtest/PPO_custom_tag_v0_0b9b4_00000_0_2024-08-09_16-29-15/checkpoint_000000"
    test_ppo(env, checkpoint)
