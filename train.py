import neat
import os
from game.game import Game
import pickle
import time
from plot import PlotReporter
import multiprocessing


def evaluate_genome(genome_data):
    net, obs = genome_data
    return net.activate(obs.flatten())


def evaluate_genomes(genomes, config):
    env = Game(num_agents=len(genomes), human_player=False)

    nets = [neat.nn.FeedForwardNetwork.create(genome, config) for _, genome in genomes]
    done = False
    observations = env.reset()

    # Set initial fitness to 0 for all genomes
    for _, genome in genomes:
        genome.fitness = 0

    # Use multiprocessing Pool to parallelize action computation
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    while not done:
        # Pair each neural network with the corresponding observation
        net_obs_pairs = zip(nets, observations)

        # Use pool.map to compute in parallel
        start = time.perf_counter()
        actions = pool.map(evaluate_genome, net_obs_pairs)
        end = time.perf_counter()
        print("NN time: ", end - start)

        # Step through the environment using the actions
        observations, rewards, done = env.step(actions)
        env.render(scale=10)

        # Update fitness for each genome
        for i, (_, genome) in enumerate(genomes):
            genome.fitness += rewards[i]

    # Close the Pool to release resources
    pool.close()
    pool.join()


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
            generation_interval=1, filename_prefix="checkpoints/neat-checkpoint-"
        )
    )
    p.add_reporter(PlotReporter())

    winner = p.run(evaluate_genomes, 100)
    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)

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

    # checkpoint_file = 'checkpoints/neat-checkpoint-41'
    # run_neat(config, checkpoint=checkpoint_file)
    run_neat(config)
