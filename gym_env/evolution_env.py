import pygame
import random
import numpy as np
from gym_env.food import Food
from gym_env.player import Player
import time
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import threading

MAP_COLOR = (255, 255, 255)
FOOD_COLOR = (0, 255, 0)
PLAYER_COLOR = (255, 0, 0)
HUMAN_PLAYER_COLOR = (255, 165, 0)
OUT_OF_BOUNDS_COLOR = (0, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
PINK = (255, 0, 255)


class EvolutionEnv(MultiAgentEnv):
    """Main game class that handles game logic."""

    def __init__(
        self,
        num_agents,
        render_mode,
        human_player,
        map_size,
        grid_size,
        food_count,
        fps,
        food_size,
        player_size,
        size_increase,
        player_speed,
        decay_rate,
        episode_length,
        scale,
        food_reward,
        kill_reward,
        kill_penalty,
        wall_penalty,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.render_mode = render_mode
        self.human_player = human_player
        self.map_size = map_size
        self.grid_size = grid_size
        self.food_count = food_count
        self.fps = fps
        self.food_size = food_size
        self.player_size = player_size
        self.size_increase = size_increase
        self.player_speed = player_speed
        self.decay_rate = decay_rate
        self.episode_length = episode_length
        self.scale = scale
        self.food_reward = food_reward
        self.kill_reward = kill_reward
        self.kill_penalty = kill_penalty
        self.wall_penalty = wall_penalty

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

        self.canvas = pygame.Surface((map_size, map_size))

        if self.render_mode == "human":
            self._init_pygame()

        if self.human_player:
            self.human_player_id = 0  # Assuming the human player is the first player
        else:
            self.human_player_id = None

    def _init_pygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.map_size * self.scale, self.map_size * self.scale))
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
            if self.human_player and agent_id == self.human_player_id:
                direction_vector = self._get_input()
            else:
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
        self.foods = [Food(position=tuple(pos), size=self.food_size, color=FOOD_COLOR) for pos in food_positions]

        player_positions = np.random.randint(0, self.map_size, size=(self.num_agents, 2))
        self.players = [
            Player(
                position=tuple(pos),
                map_size=self.map_size,
                size=self.player_size,
                speed=self.player_speed,
                color=PLAYER_COLOR if (not self.human_player or idx != self.human_player_id) else HUMAN_PLAYER_COLOR,
            )
            for idx, pos in enumerate(player_positions)
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
        reward += int(self._check_collisions_with_food(player_idx)) * self.food_reward
        reward += self._check_collisions_with_agents(player_idx)
        reward += -self.wall_penalty if self._check_wall_collision(player_idx) else 0

        self.rewards[player_idx] += reward
        if self.render_mode == "human" and player_idx == self.human_player_id:
            self.reward_text.set_text(f"Agent reward: {self.rewards[player_idx]}")

        return reward

    def render(self, scale=1):
        self.canvas.fill(MAP_COLOR)
        for food in self.foods:
            food.draw(self.canvas, 1)

        for player in self.players:
            player.draw(self.canvas, 1)

        if self.render_mode == "human":
            self.scaled_canvas = pygame.transform.scale(self.canvas, (self.map_size * scale, self.map_size * scale))
            self.screen.blit(self.scaled_canvas, (0, 0))
            food_eaten_text_surface = self.font.render(f"Food eaten: {self.food_eaten}", True, OUT_OF_BOUNDS_COLOR)
            self.screen.blit(food_eaten_text_surface, (5, 10))
            kill_count_text_surface = self.font.render(f"Kill count: {self.kill_count}", True, OUT_OF_BOUNDS_COLOR)
            self.screen.blit(kill_count_text_surface, (5, 25))
            total_reward_text_surface = self.font.render(f"Total reward: {self.total_reward}", True, OUT_OF_BOUNDS_COLOR)
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

    def _get_input(self):
        keys = pygame.key.get_pressed()
        direction_vector = pygame.Vector2(
            keys[pygame.K_RIGHT] - keys[pygame.K_LEFT],
            keys[pygame.K_DOWN] - keys[pygame.K_UP],
        )
        return direction_vector.normalize() if direction_vector.length() > 0 else direction_vector

    def _check_wall_collision(self, player_idx):
        player_position = self.players[player_idx].position
        if player_position.x <= 0 or player_position.x >= self.map_size or player_position.y <= 0 or player_position.y >= self.map_size:
            return True
        return False

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
                rewards[idx] = player_sizes[idx] * self.kill_reward
            elif collided and bigger_than_agent[idx]:
                rewards[idx] = -player_sizes[player_idx] * self.kill_penalty

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
                self.players[player_idx].size += self.size_increase
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
