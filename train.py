import neat
import os
from game.game import Game
import pickle
import time
from plot import PlotReporter
import threading


def evaluate_genome(genome_data, results, index):
    net, obs = genome_data
    action = net.activate(obs.flatten())
    results[index] = action


def evaluate_genomes(genomes, config):
    env = Game(num_agents=len(genomes), human_player=False)

    nets = [neat.nn.FeedForwardNetwork.create(genome, config) for _, genome in genomes]
    done = False
    observations = env.reset()

    # Set initial fitness to 0 for all genomes
    for _, genome in genomes:
        genome.fitness = 0

    results = [None] * len(genomes)  # Placeholder for results

    while not done:
        # Create threads for each genome's evaluation
        threads = []
        for i, net_obs_pair in enumerate(zip(nets, observations)):
            thread = threading.Thread(
                target=evaluate_genome, args=(net_obs_pair, results, i)
            )
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()
            # print('thread', thread)

        actions = results

        # Step through the environment using the actions
        observations, rewards, done = env.step(actions)
        env.render(scale=5)

        # Update fitness for each genome
        for i, (_, genome) in enumerate(genomes):
            genome.fitness += rewards[i]


def run_neat(config, checkpoint=None):
    if checkpoint is None:
        p = neat.Population(config)
    else:
        p = neat.Checkpointer.restore_checkpoint(checkpoint)
    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(
        neat.Checkpointer(
            generation_interval=10, filename_prefix="checkpoints/neat-checkpoint-"
        )
    )
    p.add_reporter(PlotReporter())

    winner = p.run(evaluate_genomes, 1000)
    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)

    print("\nBest genome:\n{!s}".format(winner))


if __name__ == "__main__":
    training_name = 'best'
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "neat_config.txt")

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    # checkpoint_file = 'checkpoints/neat-checkpoint-41'
    # run_neat(config, checkpoint=checkpoint_file)
    run_neat(config)
