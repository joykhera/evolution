import pygame
import random
import numpy as np
from env.food import Food
from env.player import Player
import time
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import threading

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


class SimpleEnv(MultiAgentEnv):
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
        self.display_size = self.map_size * self.scale
        self.rewards = np.zeros(self.num_agents)

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

        self.canvas = pygame.Surface((map_size, map_size))

        if self.render_mode == "human":
            self._init_pygame()

    def _init_pygame(self):
        pygame.init()
        pygame.display.init()
        self.screen = pygame.display.set_mode((self.display_size, self.display_size))
        pygame.display.set_caption("Evolution Environment")
        self.clock = pygame.time.Clock()
        pygame.font.init()
        self.font = pygame.font.SysFont(None, 24)

    def step(self, action_dict):
        observations = {}
        rewards = {}
        dones = {}
        infos = {}

        self.render(scale=self.scale)

        for agent_id, action in action_dict.items():
            direction_vector = self._get_action_vector(action)
            self.players[agent_id].move(direction_vector)
            self.players[agent_id].size = max(self.players[agent_id].size - self.decay_rate * self.players[agent_id].size, self.player_size)
            observations[agent_id] = self._get_agent_obs(agent_id)
            rewards[agent_id] = self._get_reward(agent_id)
            dones[agent_id] = self._is_done()

        dones["__all__"] = all(value for value in dones.values())
        self.steps += 1
        self.total_reward += sum(rewards.values())
        self.prev_step_time = time.perf_counter()
        return observations, rewards, dones, dones, infos

    def reset(self, seed=None, options=None):
        food_positions = np.random.randint(0, self.map_size, size=(self.food_count, 2))
        self.foods = [Food(position=tuple(pos), size=self.food_size, color=self.food_color) for pos in food_positions]

        player_positions = np.random.randint(0, self.map_size, size=(self.num_agents, 2))
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
        return {agent_id: self._get_agent_obs(agent_id) for agent_id in range(self.num_agents)}

    def _get_agent_obs(self, player_idx):
        player_center = self.players[player_idx].position
        half_grid = pygame.Vector2(self.grid_size // 2, self.grid_size // 2)
        top_left = player_center - half_grid
        bottom_right = player_center + half_grid

        observation = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        top_left_clamped = np.maximum(top_left, (0, 0)).astype(int)
        bottom_right_clamped = np.minimum(bottom_right, (self.map_size, self.map_size)).astype(int)

        self.render(scale=self.scale)
        self.players[player_idx].draw(self.canvas, 1, BLUE)
        for i, player in enumerate(self.players):
            if i != player_idx:
                if int(player.size) < int(self.players[player_idx].size):
                    player.draw(self.canvas, 1, PINK)
                elif int(player.size) == int(self.players[player_idx].size):
                    player.draw(self.canvas, 1, YELLOW)

        screen_pixels = pygame.surfarray.array3d(self.canvas)
        screen_pixels = np.transpose(screen_pixels, axes=(1, 0, 2))

        top_left_slice = (max(0, top_left_clamped[1]), max(0, top_left_clamped[0]))
        bottom_right_slice = (min(self.map_size, bottom_right_clamped[1]), min(self.map_size, bottom_right_clamped[0]))
        observation_slice = screen_pixels[top_left_slice[0] : bottom_right_slice[0], top_left_slice[1] : bottom_right_slice[1], :]
        observation_shape = observation_slice.shape
        top_slice = int(max(0, -top_left[1]))
        left_slice = int(max(0, -top_left[0]))

        observation[top_slice : top_slice + observation_shape[0], left_slice : left_slice + observation_shape[1], :] = observation_slice

        observation = observation / 255.0  # Normalize to [0, 1]
        observation = observation.astype(np.float32)  # Ensure dtype is float32

        return np.clip(observation, self.observation_space.low, self.observation_space.high)

    def _get_reward(self, player_idx):
        reward = 0
        reward += int(self._check_collisions_with_food(player_idx))
        reward += self._check_collisions_with_agents(player_idx)
        reward += self._check_wall_collision(player_idx)

        self.rewards[player_idx] += reward

        return reward

    def render(self, scale=1):
        self.canvas.fill(self.map_color)
        for food in self.foods:
            food.draw(self.canvas, 1)

        for player in self.players:
            player.draw(self.canvas, 1)

        if self.render_mode == "human":
            self.scaled_canvas = pygame.transform.scale(self.canvas, (self.map_size * scale, self.map_size * scale))
            self.screen.blit(self.scaled_canvas, (0, 0))
            food_eaten_text_surface = self.font.render(f"Food eaten: {self.food_eaten}", True, self.out_of_bounds_color)
            self.screen.blit(food_eaten_text_surface, (5, 10))
            kill_count_text_surface = self.font.render(f"Kill count: {self.kill_count}", True, self.out_of_bounds_color)
            self.screen.blit(kill_count_text_surface, (5, 25))
            total_reward_text_surface = self.font.render(f"Total reward: {self.total_reward}", True, self.out_of_bounds_color)
            self.screen.blit(total_reward_text_surface, (5, 40))
            self._handle_events()
            pygame.display.flip()
            self.clock.tick(self.fps)

    def _handle_events(self):
        if threading.current_thread() is threading.main_thread():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

    def _get_action_vector(self, action):
        actions = [
            pygame.Vector2(0, 0),
            pygame.Vector2(1, 0),
            pygame.Vector2(-1, 0),
            pygame.Vector2(0, -1),
            pygame.Vector2(0, 1),
        ]
        return actions[action] if action is not None else pygame.Vector2(0, 0)

    def _check_wall_collision(self, player_idx):
        player_position = self.players[player_idx].position
        if player_position.x <= 0 or player_position.x >= self.map_size or player_position.y <= 0 or player_position.y >= self.map_size:
            return -1
        return 0

    def _check_collisions_with_agents(self, player_idx):
        player_positions = np.array([p.position for p in self.players])
        player_sizes = np.array([int(p.size) for p in self.players])

        current_position = player_positions[player_idx]
        current_size = int(player_sizes[player_idx])

        distances = np.linalg.norm(player_positions - current_position, axis=1)
        radii_sum = (current_size + player_sizes) / 2

        collisions = distances < radii_sum
        collisions[player_idx] = False

        rewards = np.zeros(player_sizes.shape)
        bigger_than_agent = player_sizes > current_size
        smaller_than_agent = player_sizes < current_size

        for idx, collided in enumerate(collisions):
            if collided and smaller_than_agent[idx]:
                rewards[idx] = player_sizes[idx] * 5
            elif collided and bigger_than_agent[idx]:
                rewards[idx] = -player_sizes[player_idx] * 5

        for idx, collided in enumerate(smaller_than_agent & collisions):
            if collided:
                self.players[idx].position = pygame.Vector2(random.uniform(0, self.map_size), random.uniform(0, self.map_size))
                self.players[idx].size = self.player_size
                self.kill_count += 1
                self.players[player_idx].size += 1

        return rewards.sum()

    def _check_collisions_with_food(self, player_idx):
        for food in self.foods[:]:
            if self.players[player_idx].position.distance_to(food.position) < self.players[player_idx].size:
                food.set_position((random.randint(0, self.map_size), random.randint(0, self.map_size)))
                self.players[player_idx].size += 1
                self.food_eaten += 1
                return True

        return False

    def _is_done(self):
        return self.steps >= self.episode_length

    def observation_space_sample(self, *args, **kwargs):
        return {agent_id: self.observation_space.sample() for agent_id in self._agent_ids}

    def action_space_sample(self, *args, **kwargs):
        return {agent_id: self.action_space.sample() for agent_id in self._agent_ids}

    def observation_space_contains(self, observation):
        for agent_id, obs in observation.items():
            if not np.all((obs >= self.observation_space.low) & (obs <= self.observation_space.high)):
                return False
        return True

    def action_space_contains(self, action_dict):
        for agent_id, action in action_dict.items():
            if not (0 <= action < self.action_space.n):
                return False
        return True
