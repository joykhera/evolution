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
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch


# Register the environment
def env_creator(config):
    return ParallelPettingZooEnv(TagEnv(**config))


def get_policy_mapping_fn(agent_id, episode, **kwargs):
    if "predator" in agent_id:
        return "predator_policy"
    else:
        return "prey_policy"


def trial_name_creator(trial):
    """Creates a descriptive name for each trial based on the hyperparameter values."""
    return f"lr_{trial.config['lr']:.2e}_gamma_{trial.config['gamma']:.3f}"


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

    # Define the search space for hyperparameters
    config = (
        PPOConfig()
        .environment(env=env_name, env_config=env_config, clip_actions=True)
        .framework("torch")
        .env_runners(
            num_env_runners=model_config["num_env_runners"],  # Updated to use the new parameter name
            num_envs_per_env_runner=model_config["num_envs_per_env_runner"],  # Updated to use the new parameter name
            remote_worker_envs=True,  # Enable async sampling
        )
        .training(
            lr=tune.loguniform(1e-5, 1e-3),  # Hyperparameter tuning range
            gamma=tune.uniform(0.8, 0.999),  # Hyperparameter tuning range
            model={
                "custom_model": "custom_cnn",
            },
            # train_batch_size_per_learner=tune.choice([2000, 4000, 8000]),  # Hyperparameter tuning range
            # num_sgd_iter=tune.choice([10, 20, 30]),  # Hyperparameter tuning range
            train_batch_size_per_learner=model_config["train_batch_size_per_learner"],
            num_sgd_iter=model_config["num_sgd_iter"],
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=get_policy_mapping_fn,
        )
    )

    # Set up a scheduler and search algorithm
    scheduler = ASHAScheduler(
        metric="env_runners/episode_reward_mean",
        mode="max",
        max_t=model_config["training_iterations"],
        grace_period=20,
        reduction_factor=3,
    )
    search_alg = HyperOptSearch(metric="env_runners/episode_reward_mean", mode="max")

    print("Starting hyperparameter tuning for run", args["save_name"])
    results = tune.run(
        "PPO",
        config=config.to_dict(),
        stop={"training_iteration": model_config["training_iterations"]},
        storage_path=args["log_dir"],
        checkpoint_at_end=True,
        checkpoint_freq=0,
        callbacks=[
            WandbLoggerCallback(
                project="custom_tag",
                name=tune.sample_from(lambda spec: trial_name_creator(spec["trial"])),
                log_config=True,
            )
        ],  # Use trial name for Wandb run
        scheduler=scheduler,
        search_alg=search_alg,
        num_samples=8,  # Number of samples for hyperparameter search
        trial_name_creator=trial_name_creator,
        progress_reporter=tune.CLIReporter(max_report_frequency=model_config["max_report_frequency"]),
    )

    ray.shutdown()

    # Get the best checkpoint based on the "episode_reward_mean" metric
    best_checkpoint = results.get_best_checkpoint(
        results.get_best_trial(metric="env_runners/episode_reward_mean", mode="max"),
        metric="env_runners/episode_reward_mean",
        mode="max",
    )

    print("Saved model at:", best_checkpoint)
    return results


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

    results = train_ppo(args, model_config, env_config, env_name)
