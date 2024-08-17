import os
import ray
import optuna
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search import ConcurrencyLimiter
from optuna.samplers import TPESampler
# from optuna.integration import RaySampler
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


def train_ppo(env_config, lr, run_name):
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
        .training(lr=lr)
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
        callbacks=[WandbLoggerCallback(project="custom_tag", log_config=True, name=run_name)],
        reuse_actors=True,
    )

    ray.shutdown()
    return results


def objective(config):
    # Suggest a learning rate using Optuna
    lr = config["lr"]

    # Define your environment config
    env_config = {
        "num_prey": 2,
        "num_predators": 2,
        "prey_speed": 1,
        "predator_speed": 1,
        "map_size": 30,
        "max_steps": 100,
        "screen_size": 600,
        "grid_size": 10,
    }

    run_name = f"optuna_lr_{lr}"
    print("Trainingggggg", flush=True)
    results = train_ppo(env_config, lr, run_name)

    # Return the mean reward for evaluation
    return results.best_result["episode_reward_mean"]


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True, log_to_driver=False)
    study = optuna.create_study(direction="maximize")

    # Define the Optuna search algorithm with a ConcurrencyLimiter for parallel processing
    algo = OptunaSearch()
    algo = ConcurrencyLimiter(algo, max_concurrent=8)  # Adjust max_concurrent to control parallelism
    run_name = "optuna_parallel_search"
    # Run Tune with Optuna in parallel
    analysis = tune.run(
        objective,
        name=run_name,
        search_alg=algo,
        num_samples=8,  # Adjust as needed
        metric="mean_reward",  # Make sure this matches the metric returned in your objective function
        mode="max",
        config={"lr": tune.loguniform(1e-5, 1e-2)},  # Initial config to start
    )

    print("Best learning rate: ", study.best_params["lr"])
    print("Best trial reward: ", study.best_value)
    ray.shutdown()
