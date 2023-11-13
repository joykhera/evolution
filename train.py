import neat
import os
from game import Game
import pickle
import pygame

# Define any additional NEAT or training-specific constants and functions here


def evaluate_genome(genomes, config):
    game = Game(human_player=False)
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        game.reset()
        while not game.is_done():
            observation = game.get_observation()
            output = net.activate(observation.flatten())
            action = game.output_to_action(output)
            observation, reward, done = game.play_step(action)
            print(genome_id, action, reward, done)
        genome.fitness = reward


def run_neat(config):
    # Create the population from the configuration
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))

    # Run NEAT for a certain number of generations
    winner = p.run(evaluate_genome, 50)
    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)

    # Save or display the winning genome
    print("\nBest genome:\n{!s}".format(winner))


def test_best_network(config):
    with open("best.pickle", "rb") as f:
        winner = pickle.load(f)
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    width, height = 700, 500
    win = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Pong")
    pong = Game(win, width, height)
    pong.test_ai(winner_net)


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

    run_neat(config)
    test_best_network(config)
