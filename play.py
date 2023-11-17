import neat
import os
from game.game import Game
import time
import pygame

env = Game(num_agents=1, human_player=False)
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, "neat_config.txt")

config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    config_path,
)

done = False
observations = env.reset()
fitness = 0
episode_num = 1000


def get_wasd_actions():
    # Get the state of all keys
    keys = pygame.key.get_pressed()

    # Translate the key presses to the action array format
    # Index: 0 - None, 1 - D, 2 - A, 3 - W, 4 - S
    action = [0] * 5  # Start with no action

    if keys[pygame.K_d]:
        action[1] = 1
    if keys[pygame.K_a]:
        action[2] = 1
    if keys[pygame.K_w]:
        action[3] = 1
    if keys[pygame.K_s]:
        action[4] = 1

    # If no keys are pressed, index 0 should be 1 (representing 'none')
    if not any(action[1:]):  # If all the actions except for 'none' are 0
        action[0] = 1

    return action


for i in range(episode_num):
    while not done:
        actions = [get_wasd_actions()]
        # print(actions)

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        observations, rewards, done = env.step(actions)
        fitness += rewards[0]
        env.render(scale=10)

    print('fitness', fitness, done)
    fitness = 0
    env.reset()
    done = False
    print('reset', done)
