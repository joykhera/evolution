import neat
import os
from game.game import Game
import pickle
import time
from plot import PlotReporter

def evaluate_genomes(genomes, config):
    env = Game(num_agents=len(genomes), human_player=False)

    nets = [neat.nn.FeedForwardNetwork.create(genome, config) for _, genome in genomes]
    done = False
    observations = env.reset()

    for i, (genome_id, genome) in enumerate(genomes):
        genome.fitness = 0

    while not done:
        start = time.perf_counter()
        actions = [net.activate(obs.flatten()) for net, obs in zip(nets, observations)]
        end = time.perf_counter()
        print("Time taken: ", end - start)

        observations, rewards, done = env.step(actions)
        env.render()

        for i, (genome_id, genome) in enumerate(genomes):
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
    p.add_reporter(neat.Checkpointer(generation_interval=1, filename_prefix="checkpoints/neat-checkpoint-"))
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

    # checkpoint_file = 'neat-checkpoint-36'
    # run_neat(config, checkpoint=checkpoint_file)
    run_neat(config)
