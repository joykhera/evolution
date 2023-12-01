import pygame
import random
import numpy as np
from gym_env.food import Food
from gym_env.player import Player
import time
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# Constants

SIZE = 100
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
PINK = (255, 0, 255)
BLACK = (0, 0, 0)
FOOD_COUNT = 50
FPS = 120
FOOD_SIZE = SIZE / 50
PLAYER_SIZE = SIZE / 50
PLAYER_SIZE_INCREASE = PLAYER_SIZE / 10
PLAYER_SPEED = SIZE / 100
DECAY_RATE = 0.01
EPISODE_LENGTH = 200
SCALE = 5
GRID_SIZE = 10


class EvolutionEnv(MultiAgentEnv):
    """Main game class that handles game logic."""

    def __init__(
        self,
        num_agents=50,
        render_mode="human",
        human_player=False,
        map_size=SIZE,
        grid_size=GRID_SIZE,
        map_color=WHITE,
        food_color=GREEN,
        player_color=RED,
        out_of_bounds_color=BLACK,
        food_count=FOOD_COUNT,
        fps=FPS,
        food_size=FOOD_SIZE,
        player_size=PLAYER_SIZE,
        player_size_increase=PLAYER_SIZE_INCREASE,
        player_speed=PLAYER_SPEED,
        decay_rate=DECAY_RATE,
        episode_length=EPISODE_LENGTH,
        scale=SCALE,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.render_mode = render_mode
        self.human_player = human_player
        self.map_size = map_size
        self.grid_size = grid_size
        self.map_color = map_color
        self.food_color = food_color
        self.player_color = player_color
        self.out_of_bounds_color = out_of_bounds_color
        self.food_count = food_count
        self.fps = fps
        self.food_size = food_size
        self.player_size = player_size
        self.player_size_increase = player_size_increase
        self.player_speed = player_speed
        self.decay_rate = decay_rate
        self.episode_length = episode_length
        self.scale = scale
        self.rewards = np.zeros(self.num_agents)

        self.canvas = pygame.Surface((map_size, map_size))

        if self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((map_size * scale, map_size * scale))
            pygame.display.set_caption("Evolution Environment")
            self.clock = pygame.time.Clock()
            self.window = pygame.display.set_mode((map_size * scale, map_size * scale))
            pygame.font.init()
            self.font = pygame.font.SysFont(None, 24)
            self.scaled_canvas = pygame.Surface(
                (self.map_size * scale, self.map_size * scale)
            )

            plt.ion()  # Interactive mode on
            (self.fig, self.ax) = plt.subplots()
            self.image = self.ax.imshow(np.zeros((100, 100)), "BrBG")

            self.agent_obs_idx = 0

            # Text display for the current agent index
            self.idx_text = plt.text(
                0.02,
                0.95,
                f"Agent Index: {self.agent_obs_idx}",
                transform=self.fig.transFigure,
            )

            self.reward_text = plt.text(
                0.02,
                0.915,
                f"Agent Reward: 0",
                transform=self.fig.transFigure,
            )

            self.inc_button = Button(plt.axes([0.1, 0.85, 0.05, 0.05]), ">")
            self.dec_button = Button(plt.axes([0.04, 0.85, 0.05, 0.05]), "<")

            # Define the button's callback with a lambda function that also updates the text
            # self.button.on_clicked(
            #     lambda _: setattr(
            #         self, "agent_obs_idx", (self.agent_obs_idx + 1) % self.num_agents;
            #     )
            # )

            def increment(event):
                self.agent_obs_idx = (self.agent_obs_idx + 1) % self.num_agents
                self.idx_text.set_text(f"Agent Index: {self.agent_obs_idx}")
                # self.fig.canvas.draw_idle()

            def decrement(event):
                self.agent_obs_idx = (self.agent_obs_idx - 1) % self.num_agents
                self.idx_text.set_text(f"Agent Index: {self.agent_obs_idx}")
                # self.fig.canvas.draw_idle()

            self.inc_button.on_clicked(increment)
            self.dec_button.on_clicked(decrement)

        self.players = []
        self.foods = []
        self.running = True
        self.steps = 0
        self.prev_step_time = time.perf_counter()
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(grid_size, grid_size, 3),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(5)
        self._agent_ids = set(range(self.num_agents))
        self.food_eaten = 0
        self.total_reward = 0
        self.kill_count = 0
        self.kill_reward = 5

    def step(self, action_dict):
        """Executes a step for each agent in the environment."""
        observations = {}
        rewards = {}
        dones = {}
        infos = {}

        # self.render(scale=self.scale)

        for agent_id, action in action_dict.items():
            if self.human_player:
                direction_vector = self._get_input()
            else:
                direction_vector = self._get_action_vector(action)

            self.players[agent_id].move(direction_vector)
            if self.players[agent_id].size > self.player_size:
                self.players[agent_id].size -= (
                    self.decay_rate * self.players[agent_id].size
                )
            observations[agent_id] = self._get_agent_obs(agent_id)
            rewards[agent_id] = self._get_reward(agent_id)
            dones[agent_id] = self._is_done()

        dones["__all__"] = all(value for value in dones.values())
        self.steps += 1
        self.total_reward += sum(rewards.values())
        cur_time = time.perf_counter()
        # print("Step:", self.steps, " Step time: ", cur_time - self.prev_step_time)
        self.prev_step_time = cur_time
        return observations, rewards, dones, dones, infos

    def reset(self, seed=None, options=None):
        """Resets the game to an initial state using vectorized operations."""
        # Generate random positions for the food
        food_positions = np.random.randint(0, self.map_size, size=(self.food_count, 2))
        self.foods = [
            Food(position=tuple(pos), size=self.food_size, color=self.food_color)
            for pos in food_positions
        ]

        # Generate random positions for the players
        player_positions = np.random.randint(
            0, self.map_size, size=(self.num_agents, 2)
        )
        self.players = [
            Player(
                position=tuple(pos),
                map_size=self.map_size,
                size=self.player_size,
                speed=self.player_speed,
                color=self.player_color,
            )
            for pos in player_positions
        ]

        self.steps = 0
        self.food_eaten = 0
        self.total_reward = 0
        self.kill_count = 0
        self.rewards = np.zeros(self.num_agents)
        return self._get_obs(), {}

    def _get_obs(self):
        # Returns observation for all agents
        return {
            agent_id: self._get_agent_obs(agent_id)
            for agent_id in range(self.num_agents)
        }

    def _get_agent_obs(self, player_idx):
        player_center = self.players[player_idx].position
        half_grid = pygame.Vector2(self.grid_size // 2, self.grid_size // 2)
        top_left = player_center - half_grid
        bottom_right = player_center + half_grid

        # Initialize a black observation canvas
        observation = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)

        # Calculate the slice sizes for the observation
        top_slice = int(max(0, -top_left[1]))
        left_slice = int(max(0, -top_left[0]))

        # Clamp the top left and bottom right to be within the screen bounds
        top_left_clamped = np.maximum(top_left, (0, 0)).astype(int)
        bottom_right_clamped = np.minimum(
            bottom_right, (self.map_size, self.map_size)
        ).astype(int)

        self.render(scale=self.scale)
        self.players[player_idx].draw(self.canvas, 1, BLUE)

        for i, player in enumerate(self.players):
            if i != player_idx:
                if int(player.size) < int(self.players[player_idx].size):
                    player.draw(self.canvas, 1, PINK)
                elif int(player.size) == int(self.players[player_idx].size):
                    player.draw(self.canvas, 1, YELLOW)

        # Extract the rectangle of the screen
        screen_pixels = pygame.surfarray.array3d(self.canvas)
        screen_pixels = np.transpose(screen_pixels, axes=(1, 0, 2))

        # Copy the visible part of the screen to the observation canvas
        # Ensure that the slices do not go out of bounds
        top_left_slice = (max(0, top_left_clamped[1]), max(0, top_left_clamped[0]))
        bottom_right_slice = (
            min(self.map_size, bottom_right_clamped[1]),
            min(self.map_size, bottom_right_clamped[0]),
        )

        # Adjust the slices based on the calculated indices
        observation_slice = screen_pixels[
            top_left_slice[0] : bottom_right_slice[0],
            top_left_slice[1] : bottom_right_slice[1],
            :,
        ]
        observation_shape = observation_slice.shape

        observation[
            top_slice : top_slice + observation_shape[0],
            left_slice : left_slice + observation_shape[1],
            :,
        ] = observation_slice

        if self.render_mode == "human" and player_idx == self.agent_obs_idx:
            self.image.set_data(observation)
            plt.pause(0.001)

        # Normalize the observation by 255 to get values between 0 and 1
        return observation / 255.0

    def _get_reward(self, player_idx):
        """Returns the reward after an action."""
        reward = 0

        # Reward for eating food
        reward += int(self._check_collisions_with_food(player_idx))

        # Check for collisions with other agents
        collision_reward = self._check_collisions_with_agents(player_idx)
        reward += collision_reward

        # Check if the player is touching a wall
        wall_penalty = self._check_wall_collision(player_idx)
        reward += wall_penalty

        self.rewards[player_idx] += reward

        if self.render_mode == "human" and player_idx == self.agent_obs_idx:
            self.reward_text.set_text(f"Agent reward: {self.rewards[player_idx]}")

        # Combine the rewards
        return reward

    def render(self, scale=1):
        # Clear the canvas with a black color
        self.canvas.fill(self.map_color)
        for food in self.foods:
            food.draw(self.canvas, 1)

        for player in self.players:
            player.draw(self.canvas, 1)

        if self.render_mode == "human":
            # scaled_canvas = pygame.Surface(
            #     (self.map_size * scale, self.map_size * scale)
            # )
            # scaled_canvas.blit(
            #     pygame.transform.scale(
            #         self.canvas, (self.map_size * scale, self.map_size * scale)
            #     ),
            #     scaled_canvas.get_rect(),
            # )
            self.scaled_canvas.fill(self.map_color)
            for food in self.foods:
                food.draw(self.scaled_canvas, scale)

            for player in self.players:
                player.draw(self.scaled_canvas, scale)

            self.window.blit(self.scaled_canvas, self.scaled_canvas.get_rect())

            food_eaten_text_surface = self.font.render(
                f"Food eaten: {self.food_eaten}", True, self.out_of_bounds_color
            )
            self.window.blit(food_eaten_text_surface, (5, 10))
            kill_count_text_surface = self.font.render(
                f"Kill count: {self.kill_count}", True, self.out_of_bounds_color
            )
            self.window.blit(kill_count_text_surface, (5, 25))
            total_reward_text_surface = self.font.render(
                f"Total reward: {self.total_reward}", True, self.out_of_bounds_color
            )
            self.window.blit(total_reward_text_surface, (5, 40))

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

    def _check_wall_collision(self, player_idx):
        """Check if the player is touching a wall."""
        player_position = self.players[player_idx].position
        if (
            player_position.x <= 0
            or player_position.x >= self.map_size
            or player_position.y <= 0
            or player_position.y >= self.map_size
        ):
            return -1  # Penalty for touching a wall
        return 0

    def _check_collisions_with_agents(self, player_idx):
        """Check for collisions with other agents and adjust rewards using NumPy."""
        player_positions = np.array([p.position for p in self.players])
        player_sizes = np.array([int(p.size) for p in self.players])

        current_position = player_positions[player_idx]
        current_size = int(player_sizes[player_idx])

        # Calculate distances from the current player to all other players
        distances = np.linalg.norm(player_positions - current_position, axis=1)

        # Calculate the sum of radii (considering each player's size)
        radii_sum = (current_size + player_sizes) / 2

        # Check for collisions (distance smaller than radii sum)
        collisions = distances < radii_sum

        # Exclude the current player from collision check
        collisions[player_idx] = False

        # Assign rewards or penalties for collisions
        rewards = np.zeros(player_sizes.shape)
        bigger_than_agent = player_sizes > current_size
        smaller_than_agent = player_sizes < current_size
        # rewards[smaller_than_agent & collisions] = self.kill_reward
        # rewards[bigger_than_agent & collisions] = -self.kill_reward

        # Give the smaller player's size as the reward to the bigger player

        for idx, collided in enumerate(collisions):
            if collided and smaller_than_agent[idx]:
                rewards[idx] = player_sizes[idx] * 5
            elif collided and bigger_than_agent[idx]:
                rewards[idx] = -player_sizes[player_idx] * 5

        # Respawn any eaten agents
        for idx, collided in enumerate(smaller_than_agent & collisions):
            if collided:
                # Respawn at random location
                self.players[idx].position = pygame.Vector2(
                    random.uniform(0, self.map_size), random.uniform(0, self.map_size)
                )
                # Reset size to default
                self.players[idx].size = self.player_size
                self.kill_count += 1
                self.players[player_idx].size += 1

        # Sum up the rewards from collisions
        total_reward = rewards.sum()

        return total_reward

    def _check_collisions_with_food(self, player_idx):
        """Checks for collisions between player and food."""
        for food in self.foods[:]:
            if (
                self.players[player_idx].position.distance_to(food.position)
                < self.players[player_idx].size
            ):
                # self.foods.remove(food)
                food.set_position(
                    (random.randint(0, self.map_size), random.randint(0, self.map_size))
                )
                self.players[player_idx].size += 1
                self.food_eaten += 1
                return True

        return False

    def _is_done(self):
        """Checks if the game is finished."""
        return self.steps >= self.episode_length
        # return self.steps >= self.episode_length or self.foods == []
