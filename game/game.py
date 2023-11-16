import pygame
import random
import numpy as np
from game.food import Food
from game.player import Player

# Constants
SIZE = 50
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
FOOD_COUNT = 50
FPS = 500
FOOD_SIZE = SIZE / 50
PLAYER_SIZE = SIZE / 50
PLAYER_SIZE_INCREASE = PLAYER_SIZE / 10
PLAYER_SPEED = SIZE / 100
GRID_SIZE = 5
EPISODE_STEPS = 100
SCALE = 10

class Game:
    """Main game class that handles game logic."""

    def __init__(self, num_agents=50, human_player=True):
        pygame.init()
        pygame.display.init()
        self.human_player = human_player
        self.screen = pygame.display.set_mode((SIZE * SCALE, SIZE * SCALE))
        pygame.display.set_caption("Evolution Environment")
        self.clock = pygame.time.Clock()
        self.num_agents = num_agents
        self.players = [
            Player(
                position=(random.randint(0, SIZE), random.randint(0, SIZE)),
                map_size=SIZE,
                size=PLAYER_SIZE,
                speed=PLAYER_SPEED,
                color=RED,
            )
            for _ in range(num_agents)
        ]
        self.foods = [
            Food(map_size=SIZE, size=FOOD_SIZE, color=GREEN) for _ in range(FOOD_COUNT)
        ]
        self.running = True
        self.steps = 0
        self.window = pygame.display.set_mode((SIZE * SCALE, SIZE * SCALE))

    def handle_events(self):
        """Handles game events, such as input and quitting."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def get_action_vector(self, action):
        """Converts an action number into a direction vector."""
        actions = [
            pygame.Vector2(0, 0),  # None
            pygame.Vector2(1, 0),  # Right
            pygame.Vector2(-1, 0), # Left
            pygame.Vector2(0, -1), # Up
            pygame.Vector2(0, 1),  # Down
        ]
        return actions[action] if action is not None else pygame.Vector2(0, 0)

    def step(self, actions):
        """Executes a step for each agent in the environment."""
        observations = []
        rewards = []

        print(self.steps)
        for i, action in enumerate(actions):
            if self.human_player:
                direction_vector = self.get_input()
            else:
                max_action = self.output_to_action(action)
                direction_vector = self.get_action_vector(max_action)

            self.players[i].move(direction_vector)
            new_observation = self.get_observation()
            reward = self.get_reward(i)

            observations.append(new_observation)
            rewards.append(reward)

        done = self.is_done()
        self.steps += 1
        return observations, rewards, done

    def get_input(self):
        """Processes player input and returns a direction vector."""
        keys = pygame.key.get_pressed()
        direction_vector = pygame.Vector2(
            keys[pygame.K_RIGHT] - keys[pygame.K_LEFT],
            keys[pygame.K_DOWN] - keys[pygame.K_UP],
        )
        return (
            direction_vector.normalize()
            if direction_vector.length() > 0
            else direction_vector
        )

    def check_collisions(self, playerIdx):
        """Checks for collisions between player and food."""
        for food in self.foods[:]:
            if (
                self.players[playerIdx].position.distance_to(food.position)
                < self.players[playerIdx].size
            ):
                self.foods.remove(food)
                print("Food eaten by player ", playerIdx)
                return True

        return False

    def apply_function_to_all_agents(self, function):
        """Applies a function to all agents and return"""
        return [function(i) for i in range(len(self.players))]

    def get_observation(self):
        """Returns the entire screen as a flattened numpy array of pixel values for NEAT."""
        arr = self.render()
        return arr / 255.0

    def get_reward(self, playerIdx):
        """Returns the reward after an action."""
        return int(self.check_collisions(playerIdx))

    def is_done(self):
        """Checks if the game is finished."""
        return self.steps >= EPISODE_STEPS or self.foods == []

    def reset(self):
        """Resets the game to an initial state."""
        self.foods = [
            Food(map_size=SIZE, size=FOOD_SIZE, color=GREEN) for _ in range(FOOD_COUNT)
        ]
        self.players = [
            Player(
                position=(random.randint(0, SIZE), random.randint(0, SIZE)),
                map_size=SIZE,
                size=PLAYER_SIZE,
                speed=PLAYER_SPEED,
                color=RED,
            )
            for _ in range(self.num_agents)
        ]
        self.done = False
        observations = [self.get_observation()] * self.num_agents
        return observations

    def render(self, mode="human", scale=1):
        canvas = pygame.Surface((SIZE, SIZE))

        canvas.fill(WHITE)
        for food in self.foods:
            food.draw(canvas, 1)
        for player in self.players:
            player.draw(canvas, 1)

        scaled_canvas = pygame.Surface((SIZE * scale, SIZE * scale))
        scaled_canvas.blit(pygame.transform.scale(canvas, (SIZE * scale, SIZE * scale)), scaled_canvas.get_rect())

        if mode == "human":
            self.window.blit(scaled_canvas, scaled_canvas.get_rect())
            pygame.event.pump()
            self.clock.tick(60)

        return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))



    def close(self):
        """Closes the Pygame window."""
        pygame.quit()

    def test_ai(self, net, num_steps=1000):
        """Test the AI's performance in the game for a certain number of steps."""
        self.reset()
        pygame.display.set_caption("NEAT Test AI")
        for _ in range(num_steps):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()
            self.screen.fill(WHITE)

            observation = self.get_observation()
            output = net.activate(observation.flatten())
            self.step([output])

            self.render(scale=SCALE)
            pygame.display.flip()
            self.clock.tick(FPS)

            if self.is_done():
                break

    def output_to_action(self, output):
        return output.index(max(output))


def run_game_with_human():
    game = Game(human_player=True)
    game.run()

if __name__ == "__main__":
    run_game_with_human()
