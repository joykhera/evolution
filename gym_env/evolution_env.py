import pygame
import random
import numpy as np
from gym_env.food import Food
from gym_env.player import Player
import time
from gym import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

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
GRID_SIZE = 10


class EvolutionEnv(MultiAgentEnv):
    """Main game class that handles game logic."""

    def __init__(
        self, num_agents=50, human_player=True, map_size=SIZE, grid_size=GRID_SIZE
    ):
        pygame.init()
        pygame.display.init()
        self.map_size = map_size
        self.grid_size = grid_size
        self.human_player = human_player
        self.screen = pygame.display.set_mode((map_size * SCALE, map_size * SCALE))
        pygame.display.set_caption("Evolution Environment")
        self.clock = pygame.time.Clock()
        self.num_agents = num_agents
        self.players = []
        self.foods = []
        self.running = True
        self.steps = 0
        self.window = pygame.display.set_mode((map_size * SCALE, map_size * SCALE))
        self.canvas = pygame.Surface((map_size, map_size))
        self.prev_step_time = time.perf_counter()
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(grid_size, grid_size, 3),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(5)

    def step(self, action_dict):
        """Executes a step for each agent in the environment."""
        observations = {}
        rewards = {}
        dones = {}
        infos = {}

        for agent_id, action in action_dict.items():
            if self.human_player:
                direction_vector = self._get_input()
            else:
                direction_vector = self._get_action_vector(action)
                # max_action = self._output_to_action(action)
                # direction_vector = self._get_action_vector(max_action)

            self.players[agent_id].move(direction_vector)
            observations[agent_id] = self._get_agent_obs(agent_id)
            rewards[agent_id] = self._get_reward(agent_id)
            dones[agent_id] = self._is_done()

        dones["__all__"] = all(value for value in dones.values())
        self.steps += 1
        cur_time = time.perf_counter()
        # print("Step:", self.steps, " Step time: ", cur_time - self.prev_step_time)
        self.prev_step_time = cur_time
        self.render(scale=SCALE)
        return observations, rewards, dones, dones, infos

    def reset(self, seed=None, options=None):
        """Resets the game to an initial state."""
        self.foods = [
            Food(
                position=(
                    random.randint(0, self.map_size),
                    random.randint(0, self.map_size),
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
                    random.randint(0, self.map_size),
                    random.randint(0, self.map_size),
                ),
                map_size=self.map_size,
                size=PLAYER_SIZE,
                speed=PLAYER_SPEED,
                color=RED,
            )
            for _ in range(self.num_agents)
        ]
        self.players[0].color = BLUE
        self.steps = 0
        return self._get_obs(), {}

    def _get_obs(self):
        # Returns observation for all agents
        return {
            agent_id: self._get_agent_obs(agent_id)
            for agent_id in range(self.num_agents)
        }

    def _get_agent_obs(self, player_idx):
        player_center = self.players[player_idx].position
        top_left = player_center - pygame.Vector2(
            self.grid_size // 2, self.grid_size // 2
        )
        bottom_right = player_center + pygame.Vector2(
            self.grid_size // 2, self.grid_size // 2
        )
        # print("before clamp", player_center, top_left, bottom_right)
        # Make sure we don't go out of bounds
        top_left = self._clamp_vector(
            top_left,
            pygame.Vector2(0, 0),
            pygame.Vector2(self.map_size, self.map_size) + pygame.Vector2(1, 1),
        )
        bottom_right = self._clamp_vector(
            bottom_right,
            pygame.Vector2(0, 0),
            pygame.Vector2(self.map_size, self.map_size) - pygame.Vector2(1, 1),
        )

        # Extract the rectangle of the screen
        screen_pixels = pygame.surfarray.array3d(self.canvas)
        screen_pixels = np.transpose(np.array(screen_pixels), axes=(1, 0, 2))

        screen_pixels_scaled = pygame.surfarray.array3d(self.screen)
        screen_pixels_scaled = np.transpose(
            np.array(screen_pixels_scaled), axes=(1, 0, 2)
        )
        # print("screen_pixels.shape", screen_pixels.shape)
        # print("screen_pixels_scaled.shape", screen_pixels_scaled.shape)

        observation = screen_pixels[
            # int(top_left.x) : int(bottom_right.x), int(top_left.y) : int(bottom_right.y)
            int(top_left.y) : int(bottom_right.y),
            int(top_left.x) : int(bottom_right.x),
            :,
        ]

        # Resize the observation to a 10x10 image
        if observation.shape != (self.grid_size, self.grid_size, 3):
            # print("before", observation)
            # observation = np.resize(observation, (grid_size, grid_size, 3))
            observation = pygame.transform.smoothscale(
                pygame.surfarray.make_surface(observation),
                (self.grid_size, self.grid_size),
            )
            observation = pygame.surfarray.array3d(observation)
            observation = np.transpose(observation, axes=(1, 0, 2))
        # print("after", observation)
        return observation / 255

    def _get_reward(self, player_idx):
        """Returns the reward after an action.

        The player gets -0.1 reward for not moving.
        The player gets a larger penalty for touching a wall.
        """
        # Check if the player has eaten food
        food_reward = int(self._check_collisions(player_idx))

        # Check if the player is touching a wall
        player_position = self.players[player_idx].position
        wall_penalty = (
            -1
            if player_position.x <= 0
            or player_position.x >= self.map_size
            or player_position.y <= 0
            or player_position.y >= self.map_size
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

            # # Find the nearest food
            # nearest_food = min(
            #     self.foods, key=lambda f: player.position.distance_to(f.position)
            # )
            # # Draw a line from the player to the nearest food item using Pygame
            # pygame.draw.line(self.canvas, BLUE, player.position, nearest_food.position)
        # self.draw_rays(self.canvas, self.players[0].position, self.foods, max_distance)

        scaled_canvas = pygame.Surface((self.map_size * scale, self.map_size * scale))
        scaled_canvas.blit(
            pygame.transform.scale(
                self.canvas, (self.map_size * scale, self.map_size * scale)
            ),
            scaled_canvas.get_rect(),
        )

        if mode == "human":
            self.window.blit(scaled_canvas, scaled_canvas.get_rect())
            pygame.event.pump()
            self.clock.tick(FPS)
            pygame.display.flip()

    def _clamp_vector(self, vector, min_vector, max_vector):
        return pygame.Vector2(
            max(min_vector.x, min(vector.x, max_vector.x)),
            max(min_vector.y, min(vector.y, max_vector.y)),
        )

    def _handle_events(self):
        """Handles game events, such as input and quitting."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def _get_action_vector(self, action):
        """Converts an action number into a direction vector."""
        actions = [
            pygame.Vector2(0, 0),  # None
            pygame.Vector2(1, 0),  # Right
            pygame.Vector2(-1, 0),  # Left
            pygame.Vector2(0, -1),  # Up
            pygame.Vector2(0, 1),  # Down
        ]
        return actions[action] if action is not None else pygame.Vector2(0, 0)

    def _get_input(self):
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

    def _check_collisions(self, player_idx):
        """Checks for collisions between player and food."""
        for food in self.foods[:]:
            if (
                self.players[player_idx].position.distance_to(food.position)
                < self.players[player_idx].size
            ):
                # self.foods.remove(food)
                # print("Food eaten by player ", player_idx)
                food.set_position(
                    (random.randint(0, self.map_size), random.randint(0, self.map_size))
                )
                # print(food.position)
                return True

        return False

    def _is_done(self):
        """Checks if the game is finished."""
        return self.steps >= EPISODE_STEPS
        # return self.steps >= EPISODE_STEPS or self.foods == []

    def test_ai(self, nets, num_steps=1000):
        """Test the AI's performance in the game for a certain number of steps."""
        self.reset()
        pygame.display.set_caption("NEAT Test AI")
        for _ in range(num_steps):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()
            self.screen.fill(WHITE)

            observations = self._get_obs()

            actions = [
                net.activate(obs.flatten()) for net, obs in zip(nets, observations)
            ]

            self.step(actions)
            self.render(scale=SCALE)
            self.clock.tick(FPS)

            if self._is_done():
                break

    def _output_to_action(self, output):
        return output.index(max(output))
