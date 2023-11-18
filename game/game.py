import pygame
import random
import numpy as np
from game.food import Food
from game.player import Player
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Constants

SIZE = 100
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
FOOD_COUNT = 50
FPS = 60
FOOD_SIZE = SIZE / 50
PLAYER_SIZE = SIZE / 50
PLAYER_SIZE_INCREASE = PLAYER_SIZE / 10
PLAYER_SPEED = SIZE / 100
EPISODE_STEPS = 200
SCALE = 5


class Game:
    """Main game class that handles game logic."""

    def __init__(self, num_agents=50, human_player=True, training_name="best"):
        pygame.init()
        pygame.display.init()
        self.human_player = human_player
        self.screen = pygame.display.set_mode((SIZE * SCALE, SIZE * SCALE))
        pygame.display.set_caption("Evolution Environment")
        self.clock = pygame.time.Clock()
        self.num_agents = num_agents
        self.players = []
        self.foods = []
        self.running = True
        self.steps = 0
        self.window = pygame.display.set_mode((SIZE * SCALE, SIZE * SCALE))
        self.canvas = pygame.Surface((SIZE, SIZE))
        self.prev_step_time = time.perf_counter()
        self.training_name = training_name

    def step(self, actions):
        """Executes a step for each agent in the environment."""
        observations = []
        rewards = []

        for i, action in enumerate(actions):
            if self.human_player:
                direction_vector = self.get_input()
            else:
                max_action = self.output_to_action(action)
                direction_vector = self.get_action_vector(max_action)

            self.players[i].move(direction_vector)
            observation = self.get_observation(player_idx=i)
            reward = self.get_reward(i)

            observations.append(observation)
            rewards.append(reward)

        done = self.is_done()
        self.steps += 1
        cur_time = time.perf_counter()
        # print("Step:", self.steps, " Step time: ", cur_time - self.prev_step_time)
        self.prev_step_time = cur_time
        return observations, rewards, done

    def reset(self):
        """Resets the game to an initial state."""
        self.foods = [
            Food(
                position=(
                    random.randint(0, SIZE),
                    random.randint(0, SIZE),
                ),
                size=FOOD_SIZE,
                color=GREEN,
            )
            for _ in range(FOOD_COUNT)
        ]
        # food_per_row = int(np.sqrt(FOOD_COUNT))
        # food_per_col = FOOD_COUNT // food_per_row

        # # Calculate spacing between food items
        # x_spacing = (SIZE) // food_per_row
        # y_spacing = (SIZE) // food_per_col

        # # Create a grid of food positions
        # self.foods = []
        # for i in range(food_per_row):
        #     for j in range(food_per_col):
        #         x_position = i * x_spacing + x_spacing / 2
        #         y_position = j * y_spacing + y_spacing / 2
        #         self.foods.append(
        #             Food(
        #                 position=(x_position, y_position),
        #                 size=FOOD_SIZE,
        #                 color=GREEN,
        #             )
        #         )
        self.players = [
            Player(
                position=(
                    random.randint(0, SIZE),
                    random.randint(0, SIZE),
                ),
                map_size=SIZE,
                size=PLAYER_SIZE,
                speed=PLAYER_SPEED,
                color=RED,
            )
            for _ in range(self.num_agents)
        ]
        self.players[0].color = BLUE
        observations = [
            self.get_observation(player_idx=i) for i in range(self.num_agents)
        ] * self.num_agents
        self.steps = 0
        return observations

    def get_observation(self, player_idx):
        """
        Get the observation for a player.

        :param player_idx: Index of the player for which to get the observation
        :return: An array of observations including the player's normalized position,
                the normalized position of the nearest food item,
                and the distances to each wall from the player's position.
        """
        player_position = self.players[player_idx].position

        # Normalize player's position
        normalized_player_position = (
            player_position.x / SIZE,
            player_position.y / SIZE,
        )

        # Find the nearest food and its normalized position
        nearest_food_position = min(
            self.foods, key=lambda food: player_position.distance_to(food.position)
        ).position

        normalized_nearest_food_position = (
            nearest_food_position.x / SIZE,
            nearest_food_position.y / SIZE,
        )

        # Calculate distances to walls
        distance_to_top_wall = player_position.y / SIZE
        distance_to_bottom_wall = (SIZE - player_position.y) / SIZE
        distance_to_left_wall = player_position.x / SIZE
        distance_to_right_wall = (SIZE - player_position.x) / SIZE

        # Find the distance to the closest wall
        distance_to_closest_wall = (
            min(
                distance_to_top_wall,
                distance_to_bottom_wall,
                distance_to_left_wall,
                distance_to_right_wall,
            )
            / SIZE
        )

        # Construct the observation array
        observation = np.array(
            [
                normalized_player_position[0],
                normalized_player_position[1],  # Player's position
                normalized_player_position[0] - normalized_nearest_food_position[0],
                normalized_player_position[1] - normalized_nearest_food_position[1],
                # distance_to_closest_wall
                distance_to_top_wall,
                distance_to_bottom_wall,
                distance_to_left_wall,
                distance_to_right_wall,  # Distances to walls
            ]
        )
        # if player_idx == 0:
        #     print("Observation: ", observation)
        return observation

    def get_reward(self, player_idx):
        """Returns the reward after an action.

        The player gets -0.1 reward for not moving.
        The player gets a larger penalty for touching a wall.
        """
        # Check if the player has eaten food
        food_reward = int(self.check_collisions(player_idx))

        # Check if the player is touching a wall
        player_position = self.players[player_idx].position
        wall_penalty = (
            -1
            if player_position.x <= 0
            or player_position.x >= SIZE
            or player_position.y <= 0
            or player_position.y >= SIZE
            else 0
        )

        # Check if the player did not move (assuming you have a way to check the previous position)
        move_penalty = 0
        if (
            self.players[player_idx].position
            == self.players[player_idx].previous_position
        ):
            move_penalty = -0.1

        # Combine the rewards
        reward = food_reward + wall_penalty + move_penalty
        return reward

    def render(self, mode="human", scale=1):
        # Clear the canvas with a black color
        self.canvas.fill(WHITE)
        for food in self.foods:
            food.draw(self.canvas, 1)

        for player in self.players:
            player.draw(self.canvas, 1)

            # Find the nearest food
            nearest_food = min(
                self.foods, key=lambda f: player.position.distance_to(f.position)
            )
            # Draw a line from the player to the nearest food item using Pygame
            pygame.draw.line(self.canvas, BLUE, player.position, nearest_food.position)
        # self.draw_rays(self.canvas, self.players[0].position, self.foods, max_distance)

        scaled_canvas = pygame.Surface((SIZE * scale, SIZE * scale))
        scaled_canvas.blit(
            pygame.transform.scale(self.canvas, (SIZE * scale, SIZE * scale)),
            scaled_canvas.get_rect(),
        )

        if mode == "human":
            self.window.blit(scaled_canvas, scaled_canvas.get_rect())
            pygame.event.pump()
            self.clock.tick(FPS)
            pygame.display.flip()

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
            pygame.Vector2(-1, 0),  # Left
            pygame.Vector2(0, -1),  # Up
            pygame.Vector2(0, 1),  # Down
        ]
        return actions[action] if action is not None else pygame.Vector2(0, 0)

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

    def check_collisions(self, player_idx):
        """Checks for collisions between player and food."""
        for food in self.foods[:]:
            if (
                self.players[player_idx].position.distance_to(food.position)
                < self.players[player_idx].size
            ):
                # self.foods.remove(food)
                # print("Food eaten by player ", player_idx)
                food.set_position((random.randint(0, SIZE), random.randint(0, SIZE)))
                # print(food.position)
                return True

        return False

    def apply_function_to_all_agents(self, function):
        """Applies a function to all agents and return"""
        return [function(i) for i in range(len(self.players))]

    def _clamp_vector(self, vector, min_vector, max_vector):
        return pygame.Vector2(
            max(min_vector.x, min(vector.x, max_vector.x)),
            max(min_vector.y, min(vector.y, max_vector.y)),
        )

    def is_done(self):
        """Checks if the game is finished."""
        return self.steps >= EPISODE_STEPS
        # return self.steps >= EPISODE_STEPS or self.foods == []

    def close(self):
        """Closes the Pygame window."""
        pygame.quit()

    def test_ai(self, nets, num_steps=1000):
        """Test the AI's performance in the game for a certain number of steps."""
        self.reset()
        pygame.display.set_caption("NEAT Test AI")
        for _ in range(num_steps):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()
            self.screen.fill(WHITE)

            # observation = self.get_observation()
            observations = [
                self.get_observation(player_idx=i) for i in range(self.num_agents)
            ]

            actions = [
                net.activate(obs.flatten()) for net, obs in zip(nets, observations)
            ]

            self.step(actions)
            self.render(scale=SCALE)
            self.clock.tick(FPS)

            if self.is_done():
                break

    def output_to_action(self, output):
        return output.index(max(output))


if __name__ == "__main__":
    game = Game(human_player=True)
    game.run()
