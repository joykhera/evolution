import numpy as np
from pettingzoo.utils import ParallelEnv
from gymnasium import spaces
import pygame


class TagEnv(ParallelEnv):
    metadata = {"render.modes": ["human"], "name": "custom_tag_v0"}

    def __init__(self):
        self.agents = ["agent_0", "adversary_0", "adversary_1"]
        self.possible_agents = self.agents[:]
        self.agent_positions = np.zeros((self.num_agents, 2))
        self.grid_size = 50
        self.max_steps = 200
        self.current_step = 0

        # Initialize pygame screen
        self.screen_size = 600
        self.scale = self.screen_size // self.grid_size
        self.screen = None

        # Define observation and action spaces
        self.observation_spaces = {agent: spaces.Box(low=0, high=self.grid_size, shape=(2,), dtype=np.float32) for agent in self.agents}
        self.action_spaces = {agent: spaces.Discrete(5) for agent in self.agents}  # 4 directions + no action

    def reset(self, seed=None, return_info=False, options=None):
        self.current_step = 0
        self.agents = self.possible_agents[:]
        self.agent_positions = np.random.rand(self.num_agents, 2) * self.grid_size
        observations = {agent: self.agent_positions[i] for i, agent in enumerate(self.agents)}
        return observations, {}

    def step(self, actions):
        self.current_step += 1
        rewards = {}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        # Update positions based on actions
        for i, agent in enumerate(self.agents):
            if actions[agent] == 1:  # move left
                self.agent_positions[i][0] = max(self.agent_positions[i][0] - 1, 0)
            elif actions[agent] == 2:  # move right
                self.agent_positions[i][0] = min(self.agent_positions[i][0] + 1, self.grid_size)
            elif actions[agent] == 3:  # move down
                self.agent_positions[i][1] = max(self.agent_positions[i][1] - 1, 0)
            elif actions[agent] == 4:  # move up
                self.agent_positions[i][1] = min(self.agent_positions[i][1] + 1, self.grid_size)

        # Compute rewards
        agent_pos = self.agent_positions[0]
        adversary_positions = self.agent_positions[1:]

        # Calculate distances
        distances = np.array([np.linalg.norm(agent_pos - pos) for pos in adversary_positions])
        min_distance = distances.min()

        # Maximum possible distance in the grid (from one corner to the opposite corner)
        max_possible_distance = np.sqrt(2) * self.grid_size

        # Agent reward: maximize distance from the closest adversary
        rewards[self.agents[0]] = min_distance / max_possible_distance

        # Adversary rewards: max possible distance - current distance to the agent
        for i, distance in enumerate(distances):
            rewards[self.agents[i + 1]] = (max_possible_distance - distance) / max_possible_distance

        # Implement termination condition: if min_distance < 1 or max steps reached
        if min_distance < 1:
            terminations = {agent: True for agent in self.agents}
        if self.current_step >= self.max_steps:
            truncations = {agent: True for agent in self.agents}

        observations = {agent: self.agent_positions[i] for i, agent in enumerate(self.agents)}
        return observations, rewards, terminations, truncations, infos

    def render(self, mode="human"):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))

        self.screen.fill((255, 255, 255))

        # Draw agent
        pygame.draw.circle(self.screen, (0, 255, 0), (self.agent_positions[0] * self.scale).astype(int), self.scale)

        # Draw adversaries
        for pos in self.agent_positions[1:]:
            pygame.draw.circle(self.screen, (255, 0, 0), (pos * self.scale).astype(int), self.scale)

        pygame.display.flip()

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
