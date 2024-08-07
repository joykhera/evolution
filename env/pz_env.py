import pygame
import functools
import random
import numpy as np
from env.food import Food
from env.player import Player
import time
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector
from pettingzoo.utils import wrappers
from pettingzoo.utils import parallel_to_aec, wrappers
import threading

MAP_COLOR = (255, 255, 255)
FOOD_COLOR = (0, 255, 0)
PLAYER_COLOR = (255, 0, 0)
HUMAN_PLAYER_COLOR = (255, 165, 0)
OUT_OF_BOUNDS_COLOR = (0, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
PINK = (255, 0, 255)


def env(config):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    env = EvolutionEnv(**config)
    env = parallel_to_aec(env)
    # env = raw_env(**config)
    env = wrappers.OrderEnforcingWrapper(env)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    return env


class EvolutionEnv(ParallelEnv):
    # class raw_env(AECEnv):
    """Main game class that handles game logic."""

    metadata = {"render.modes": ["human", "rgb_array"], "name": "evolution_v0"}

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
        max_speed,
        max_boost,
        decay_rate,
        episode_length,
        scale,
        food_reward,
        kill_reward,
        kill_penalty,
        wall_penalty,
    ):
        super().__init__()
        self.num_agentss = num_agents
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
        self.max_speed = max_speed
        self.max_boost = max_boost
        self.decay_rate = decay_rate
        self.episode_length = episode_length
        self.scale = scale
        self.food_reward = food_reward
        self.kill_reward = kill_reward
        self.kill_penalty = kill_penalty
        self.wall_penalty = wall_penalty

        self.agents = list(range(self.num_agentss))
        # self.agents = [f"player_{i}" for i in range(self.num_agentss)]
        # self.agent_name_mapping = {agent: i for i, agent in enumerate(self.agents)}
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.rewards = None
        self.dones = None
        self.infos = None
        self.observations = None

        self.players = []
        self.foods = []
        self.steps = 0
        self.prev_step_time = time.perf_counter()

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

    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent=None):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return spaces.Box(
            low=0,
            high=1,
            shape=(self.grid_size, self.grid_size, 3),
            dtype=np.float32,
        )

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Discrete(5)

    # def possible_agents(self):
    #     return self.agents

    def _init_pygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.map_size * self.scale, self.map_size * self.scale))
        pygame.display.set_caption("Evolution Environment")
        self.clock = pygame.time.Clock()
        pygame.font.init()
        self.font = pygame.font.SysFont(None, 24)

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

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the following attributes
        - agents
        - rewards
        - _cumulative_rewards
        - terminations
        - truncations
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.
        Here it sets up the state dictionary which is used by step() and the observations dictionary which is used by step() and observe()
        """
        food_positions = np.random.randint(0, self.map_size, size=(self.food_count, 2))
        self.foods = [Food(position=tuple(pos), size=self.food_size, color=FOOD_COLOR) for pos in food_positions]
        player_positions = np.random.randint(0, self.map_size, size=(self.num_agentss, 2))
        self.players = [
            Player(
                position=tuple(pos),
                map_size=self.map_size,
                size=self.player_size,
                speed=self.player_speed,
                color=PLAYER_COLOR if (not self.human_player or idx != self.human_player_id) else HUMAN_PLAYER_COLOR,
                boost_color=BLUE,
                max_speed=self.max_speed,
                max_boost=self.max_boost,
            )
            for idx, pos in enumerate(player_positions)
        ]
        self.agents = list(range(self.num_agentss))
        self.steps = 0
        self.food_eaten = 0
        self.total_reward = 0
        self.kill_count = 0
        self.rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        # observations = {agent: None for agent in self.agents}
        observations = self._get_obs()
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, action_dict):
        """
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - terminations
        - truncations
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        """
        observations = {}
        rewards = {}
        dones = {}
        infos = {}

        self.render(scale=self.scale)

        for agent_id, action in action_dict.items():
            # direction, boost = action
            direction = action
            if self.human_player and agent_id == self.human_player_id:
                direction_vector = self._get_input()
            else:
                direction_vector = self._get_action_vector(direction)
            self.players[agent_id].move(direction_vector, 0)
            self.players[agent_id].size = max(self.players[agent_id].size - self.decay_rate * self.players[agent_id].size, self.player_size)
            observations[agent_id] = self._get_agent_obs(agent_id)
            dones[agent_id] = self._is_done()
            if not dones[agent_id]:
                rewards[agent_id] = self._get_reward(agent_id)
            else:
                rewards[agent_id] = 0
                self.agents.remove(agent_id)

        dones["__all__"] = all(dones.values())
        self.steps += 1
        self.total_reward += sum(rewards.values())
        self.prev_step_time = time.perf_counter()
        return observations, rewards, dones, dones, infos

    def _get_obs(self):
        return {agent: self._get_agent_obs(idx) for idx, agent in enumerate(self.agents)}

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
        # print('observation type', np.clip(observation, self.observation_space.low, self.observation_space.high).dtype)
        # return {
        #     "visual": np.clip(observation, self.observation_space.low, self.observation_space.high),
        #     "boost_info": np.array([self.players[player_idx].cur_boost / self.max_boost]),
        # }
        return np.clip(observation, self.observation_space().low, self.observation_space().high)

    def _get_reward(self, player_idx):
        reward = 0
        reward += int(self._check_collisions_with_food(player_idx)) * self.food_reward
        reward += self._check_collisions_with_agents(player_idx)
        reward += -self.wall_penalty if self._check_wall_collision(player_idx) else 0

        self.rewards[player_idx] += reward
        if self.render_mode == "human" and player_idx == self.human_player_id:
            self.reward_text.set_text(f"Agent reward: {self.rewards[player_idx]}")

        return reward

    def _get_action_vector(self, direction):
        actions = [
            pygame.Vector2(0, 0),
            pygame.Vector2(1, 0),
            pygame.Vector2(-1, 0),
            pygame.Vector2(0, -1),
            pygame.Vector2(0, 1),
        ]
        return actions[direction] if direction is not None else pygame.Vector2(0, 0)

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
        same_size_as_agent = player_sizes == current_size

        for idx, collided in enumerate(collisions):
            if collided and smaller_than_agent[idx]:
                rewards[idx] = player_sizes[idx] * self.kill_reward
            elif collided and bigger_than_agent[idx]:
                rewards[idx] = -player_sizes[player_idx] * self.kill_penalty
            elif collided and same_size_as_agent[idx]:
                rewards[idx] = -1

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
