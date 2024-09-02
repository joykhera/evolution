import ray
from pprint import pprint
from argparser import parse_args
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.air.integrations.wandb import WandbLoggerCallback
from deap import base, creator, tools, algorithms
import random
import numpy as np
from env import TagEnv
import torch
from ray.rllib.env import ParallelPettingZooEnv
from ray.tune.registry import register_env
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


def train_ppo_neuroevolution(args, model_config, env_config, env_name):
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
            num_env_runners=model_config["num_env_runners"],
            num_envs_per_env_runner=model_config["num_envs_per_env_runner"],
            remote_worker_envs=True,
        )
        .training(
            lr=model_config["learning_rate"],
            gamma=model_config["gamma"],
            model={"custom_model": "custom_cnn"},
            train_batch_size_per_learner=model_config["train_batch_size_per_learner"],
            num_sgd_iter=model_config["num_sgd_iter"],
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=get_policy_mapping_fn,
        )
    )

    def eval_genomes(genomes, config):
        results = []
        for genome in genomes:
            # Set the model weights using the genome
            set_model_weights_from_genome(genome, model_config)

            # Run training with the current configuration
            result = tune.run(
                "PPO",
                config=config.to_dict(),
                stop={"training_iteration": model_config["training_iterations"]},
                storage_path=args["log_dir"],
                checkpoint_at_end=False,
                reuse_actors=True,
                progress_reporter=tune.CLIReporter(max_report_frequency=model_config["max_report_frequency"]),
                verbose=0,
            )

            # Get the fitness score based on the "episode_reward_mean" metric
            fitness = result.results.get_best_trial(metric="episode_reward_mean", mode="max").last_result["episode_reward_mean"]
            results.append(fitness)

        return results

    # Set up the DEAP framework for neuroevolution
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, -1, 1)  # Example of how to initialize weights
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=model_config["num_weights"])
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", eval_genomes, config=config)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=model_config["population_size"])

    print("Starting neuroevolution training for run:", args["save_name"])

    # Run the neuroevolution algorithm
    for generation in range(model_config["generations"]):
        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        population = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, population))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < model_config["cxpb"]:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < model_config["mutpb"]:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        population[:] = offspring

        # Log or save results for each generation if necessary
        best_individual = tools.selBest(population, 1)[0]
        print(f"Generation {generation}: Best fitness = {best_individual.fitness.values[0]}")

    best_individual = tools.selBest(population, 1)[0]
    print("Best individual is:", best_individual)
    print("Best fitness:", best_individual.fitness.values[0])

    ray.shutdown()
    return best_individual


def set_model_weights_from_genome(genome, model_config):
    """
    Set the model weights based on the provided genome.

    Args:
        genome (list of floats): A flat list of numbers representing the model's weights.
        model_config (dict): Configuration containing the model and other necessary parameters.
    """
    # Assuming 'model' is passed in the model_config dictionary
    model = model_config["model"]
    current_index = 0

    # Iterate over the model's parameters and set them according to the genome
    for param in model.parameters():
        # Calculate the number of weights needed for this layer
        param_size = param.numel()

        # Extract the corresponding slice from the genome
        genome_slice = genome[current_index : current_index + param_size]

        # Convert the slice to a tensor and reshape it to match the parameter's shape
        param_tensor = torch.tensor(genome_slice, dtype=param.dtype).view(param.shape)

        # Set the parameter in the model
        with torch.no_grad():
            param.copy_(param_tensor)

        # Move to the next section of the genome
        current_index += param_size

    # Ensure that the genome is fully utilized
    assert current_index == len(genome), "Genome size does not match the total number of model parameters."


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
    train_ppo_neuroevolution(args, model_config, env_config, env_name)
