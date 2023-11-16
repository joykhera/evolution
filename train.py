import neat
import os
from game.game import Game
import pickle
import pygame
import numpy as np
import time
from plot import PlotReporter

# Define any additional NEAT or training-specific constants and functions here


def evaluate_genomes(genomes, config):
    # Create an environment with the same number of agents as genomes
    env = Game(num_agents=len(genomes), human_player=False)

    # Create neural networks for each genome
    nets = [neat.nn.FeedForwardNetwork.create(genome, config) for _, genome in genomes]
    done = False
    # Reset the environment and run the simulation
    observations = env.reset()
    while not done:
        # Get actions for all agents from their respective networks
        start = time.perf_counter()
        actions = [net.activate(obs.flatten()) for net, obs in zip(nets, observations)]
        end = time.perf_counter()
        print("Time taken: ", end - start)
        # print("observations: ", [np.unique(observation, return_counts=True) for observation in observations])
        # print("Observations:")
        # for i, observation in enumerate(observations, 1):
        #     print(f"Observation {i}:")
        #     unique_elements, counts = np.unique(observation, return_counts=True)
        #     for element, count in zip(unique_elements, counts):
        #         print(f"Number {element}: {count} times.")
        #     print()

        # Step the environment
        observations, rewards, done = env.step(actions)
        env.render()

        # Assign fitness to genomes based on rewards
        # print(len(genomes), len(rewards))
        for i, (genome_id, genome) in enumerate(genomes):
            if genome.fitness is None:
                genome.fitness = rewards[i]
            else:
                genome.fitness += rewards[i]
            # print(genome.fitness)

        if max(rewards) > 0:
            # print(rewards, list(map(lambda x: x.fitness, genomes)))
            print(list(map(lambda x: x[1].fitness, genomes)))
            # print(rewards, genomes)


def run_neat(config, checkpoint=None):
    if checkpoint is None:
        p = neat.Population(config)
    else:
        p = neat.Checkpointer.restore_checkpoint(checkpoint)
    # Create the population from the configuration
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))
    p.add_reporter(PlotReporter())

    # Run NEAT for a certain number of generations
    winner = p.run(evaluate_genomes, 49)
    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)

    # Save or display the winning genome
    print("\nBest genome:\n{!s}".format(winner))

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "neat_config.txt")

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    checkpoint_file = 'neat-checkpoint-36'
    # run_neat(config, checkpoint=checkpoint_file)
    run_neat(config)
