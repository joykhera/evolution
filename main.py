import ray
from pprint import pprint
from argparser import parse_args
from env import TagEnv
from ray import tune
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune.registry import register_env
from ray.rllib.env import ParallelPettingZooEnv
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.visionnet import VisionNetwork

# Register the environment
def env_creator(config):
    return ParallelPettingZooEnv(TagEnv(**config))

def get_policy_mapping_fn(agent_id, episode, **kwargs):
    if "predator" in agent_id:
        return "predator_policy"
    else:
        return "prey_policy"


def train_ppo(args, model_config, env_config, env_name):
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
        .environment(env=env_name, env_config=env_config, clip_actions=True)
        .framework("torch")
        .env_runners(
            rollout_fragment_length="auto",
            num_env_runners=model_config["num_env_runners"],  # Updated to use the new parameter name
            num_envs_per_env_runner=model_config["num_envs_per_env_runner"],  # Updated to use the new parameter name
            remote_worker_envs=True,  # Enable async sampling
        )
        .training(
            lr=model_config["learning_rate"],
            model={
                "custom_model": "custom_cnn",
            },
            train_batch_size_per_learner=model_config["train_batch_size_per_learner"],  # New setting for batch size per learner
            num_sgd_iter=model_config["num_sgd_iter"],
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=get_policy_mapping_fn,
        )
    )
    print("Starting training for run:", args['save_name'])
    results = tune.run(
        "PPO",
        name=args["save_name"],
        config=config.to_dict(),
        stop={"training_iteration": model_config["training_iterations"]},
        storage_path=args["log_dir"],
        checkpoint_at_end=True,
        checkpoint_freq=0,
        callbacks=[
            WandbLoggerCallback(
                project="custom_tag",
                name=args["save_name"],
                save_code=False,
                log_config_interval=10,
            )
        ],
        # verbose=0,
        reuse_actors=True,
        # progress_reporter=tune.CLIReporter(max_report_frequency=100),
    )

    ray.shutdown()
    print('Saved model at:', results.get_best_checkpoint(results.get_best_trial(metric="episode_reward_mean"), metric="episode_reward_mean"))
    return results

def test_ppo(args, env_config):
    ray.init(local_mode=True, ignore_reinit_error=True, log_to_driver=False)
    trainer = Algorithm.from_checkpoint(args['checkpoint'])
    env_config["render_mode"] = "human"
    env = TagEnv(**env_config)
    print('model initialized')

    for episode in range(args['test_episodes']):
        episode_predator_reward = 0
        episode_prey_reward = 0
        done = False
        observations, infos = env.reset()
        while not done:
            actions = {agent: trainer.compute_single_action(observations[agent], policy_id=get_policy_mapping_fn(agent, None)) for agent in env.agents}
            observations, rewards, terminations, truncations, infos = env.step(actions)
            episode_prey_reward += sum(rewards[prey] for prey in rewards if "prey" in prey) / env_config["prey_count"]
            episode_predator_reward += sum(rewards[predator] for predator in rewards if "predator" in predator) / env_config["predator_count"]
            done = any(terminations.values()) or any(truncations.values())
        print(f"Episode {episode}, predator reward: {episode_predator_reward}, prey reward: {episode_prey_reward}")

    ray.shutdown()


if __name__ == "__main__":
    args, model_config, env_config = parse_args()
    print("args:")
    pprint(args)
    print("model_config:")
    pprint(model_config)
    print("env_config:")
    pprint(env_config)
    env_name = "custom_tag_v0"
    register_env(env_name, env_creator)
    ModelCatalog.register_custom_model("custom_cnn", VisionNetwork)
    print("Environment registered:", env_name)

    if args['train']:
        results = train_ppo(args, model_config, env_config, env_name)
    elif args['test']:
        test_ppo(args, env_config)
    else:
        print("Please provide either --train or --test flag.")
