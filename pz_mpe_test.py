import os
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.air.integrations.mlflow import MLflowLoggerCallback
from pettingzoo.mpe import simple_tag_v3
from ray.rllib.env import ParallelPettingZooEnv
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig


# Register the environment
def env_creator(config):
    return ParallelPettingZooEnv(simple_tag_v3.parallel_env(render_mode="rgb_array"))


register_env("simple_tag_v3", env_creator)


def get_policy_mapping_fn(agent_id, episode, **kwargs):
    if "adversary" in agent_id:
        return "adversary_policy"
    else:
        return "agent_policy"


def train_ppo():
    ray.init(ignore_reinit_error=True, log_to_driver=False)

    config = (
        PPOConfig()
        .environment(env="simple_tag_v3", clip_actions=True)
        .framework("torch")
        .rollouts(rollout_fragment_length="auto")
        .multi_agent(
            policies={
                "adversary_policy": (None, simple_tag_v3.parallel_env().observation_space("adversary_0"), simple_tag_v3.parallel_env().action_space("adversary_0"), {}),
                "agent_policy": (None, simple_tag_v3.parallel_env().observation_space("agent_0"), simple_tag_v3.parallel_env().action_space("agent_0"), {}),
            },
            policy_mapping_fn=get_policy_mapping_fn,
        )
    )
    config.num_env_runners = 9
    print('Starting training for run:', run_name)
    results = tune.run(
        "PPO",
        name=run_name,
        config=config.to_dict(),
        stop={"training_iteration": 500},
        storage_path=os.path.join(os.getcwd(), "training"),
        checkpoint_at_end=True,
        checkpoint_freq=0,
        callbacks=[
            MLflowLoggerCallback(
                experiment_name=run_name,
                save_artifact=True,
            )
        ],
        verbose=0,
    )

    ray.shutdown()
    return results


def test_ppo(checkpoint_path):
    ray.init(ignore_reinit_error=True, log_to_driver=False)
    trainer = Algorithm.from_checkpoint(checkpoint_path)

    env = simple_tag_v3.parallel_env(render_mode="human")

    for episode in range(10):
        episode_adversary_reward = 0
        episode_agent_reward = 0
        done = False
        observations, _ = env.reset()  # This should return a dictionary of observations
        while not done:
            actions = {agent: trainer.compute_single_action(observations[agent], policy_id=get_policy_mapping_fn(agent, None)) for agent in env.agents}
            obs, rewards, dones, _, _ = env.step(actions)
            # print(actions, rewards)
            episode_adversary_reward += rewards["adversary_0"] + rewards["adversary_1"] + rewards["adversary_2"]
            episode_agent_reward += rewards["agent_0"]
            done = all(dones.values())
        print(f"Episode {episode}, adversary reward: {episode_adversary_reward}, agent reward: {episode_agent_reward}")

    ray.shutdown()


if __name__ == "__main__":
    run_name = 'test500'
    # results = train_ppo()
    # checkpoint = results.get_best_checkpoint(results.get_last_trial(), mode="max")
    # test_ppo(checkpoint)
    test_ppo("training/test200/PPO_simple_tag_v3_f4637_00000_0_2024-08-09_11-27-57/checkpoint_000000")
