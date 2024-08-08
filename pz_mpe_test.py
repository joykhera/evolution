import os
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.air.integrations.mlflow import MLflowLoggerCallback
from pettingzoo.mpe import simple_tag_v3
from pettingzoo.utils import parallel_to_aec
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
        .rollouts(rollout_fragment_length=200)
        .resources(num_cpus=9)
        .multi_agent(
            policies={
                "adversary_policy": (None, simple_tag_v3.parallel_env().observation_space("adversary_0"), simple_tag_v3.parallel_env().action_space("adversary_0"), {}),
                "agent_policy": (None, simple_tag_v3.parallel_env().observation_space("agent_0"), simple_tag_v3.parallel_env().action_space("agent_0"), {}),
            },
            policy_mapping_fn=get_policy_mapping_fn,
        )
    )

    results = tune.run(
        "PPO",
        config=config.to_dict(),
        stop={"training_iteration": 100},
        storage_path=os.path.join(os.getcwd(), "training"),
        checkpoint_at_end=True,
        callbacks=[MLflowLoggerCallback()],
    )

    ray.shutdown()
    return results


def test_ppo(checkpoint_path):
    ray.init(ignore_reinit_error=True, log_to_driver=False)
    trainer = Algorithm.from_checkpoint(checkpoint_path)

    env = simple_tag_v3.parallel_env(render_mode="human")

    for episode in range(10):
        episode_reward = 0
        done = False
        observations, _ = env.reset()  # This should return a dictionary of observations
        while not done:
            actions = {agent: trainer.compute_single_action(observations[agent], policy_id=get_policy_mapping_fn(agent, None)) for agent in env.agents}
            obs, rewards, dones, _, _ = env.step(actions)
            # print(actions, rewards)
            episode_reward += sum(rewards.values())
            done = all(dones.values())
        print(f"Episode {episode} reward: {episode_reward}")

    ray.shutdown()


if __name__ == "__main__":
    # results = train_ppo()
    # checkpoint = results.get_best_checkpoint(results.get_last_trial(), mode="max")
    # test_ppo(checkpoint)
    test_ppo('training/PPO_2024-08-08_13-35-47/PPO_simple_tag_v3_a5fdb_00000_0_2024-08-08_13-35-47/checkpoint_000000')
