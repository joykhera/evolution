import neat
import pickle
from game.game import Game
import os


def test_best_network(config, num_agents=1):
    with open("best.pickle", "rb") as f:
        winner = pickle.load(f)
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    winner_nets = [winner_net for _ in range(num_agents)]

    game = Game(num_agents=num_agents, human_player=False)
    game.test_ai(winner_nets)

    # game = Game(num_agents=1, human_player=False)
    # game.test_ai(winner_net)


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

    test_best_network(config, num_agents=20)
