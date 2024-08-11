import numpy as np
from pettingzoo.utils import ParallelEnv
from gymnasium import spaces
import pygame


class TagEnv(ParallelEnv):
    metadata = {"render.modes": ["human"], "name": "custom_tag_v0"}

    def __init__(self, num_prey=1, num_predators=2, grid_size=50, max_steps=400, screen_size=600, render_mode=None):
        self.num_prey = num_prey
        self.num_predators = num_predators
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.screen_size = screen_size
        self.render_mode = render_mode

        self.agents = [f"prey_{i}" for i in range(num_prey)] + [f"predator_{i}" for i in range(num_predators)]
        self.possible_agents = self.agents[:]
        self.agent_positions = np.zeros((len(self.agents), 2))
        self.current_step = 0

        # Initialize pygame screen
        self.scale = self.screen_size // self.grid_size
        self.screen = None if self.render_mode != "human" else pygame.display.set_mode((self.screen_size, self.screen_size))

        # Define observation and action spaces
        self.observation_spaces = {
            agent: (
                spaces.Box(low=-1, high=1, shape=(2 * self.num_predators,), dtype=np.float32)  # Two values (x and y) per predator for each prey
                if "prey" in agent
                else spaces.Box(low=-1, high=1, shape=(2 * self.num_prey,), dtype=np.float32)  # Two values (x and y) per prey for each predator
            )
            for agent in self.agents
        }
        self.action_spaces = {agent: spaces.Discrete(5) for agent in self.agents}  # 4 directions + no action

        # Track previous distances for rewards
        self.prev_distances = None

    def reset(self, seed=None, return_info=False, options=None):
        self.current_step = 0
        self.agents = self.possible_agents[:]
        self.agent_positions = np.random.rand(len(self.agents), 2) * self.grid_size

        # Initialize previous distances
        prey_positions = self.agent_positions[: self.num_prey]
        predator_positions = self.agent_positions[self.num_prey :]
        self.prev_distances = np.array([[np.linalg.norm(prey_pos - pred_pos) for pred_pos in predator_positions] for prey_pos in prey_positions])

        observations = self.get_observations()
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

        if self.render_mode == "human":
            self.render()

        # Compute rewards
        prey_positions = self.agent_positions[: self.num_prey]
        predator_positions = self.agent_positions[self.num_prey :]

        # Calculate current distances
        distances = np.array([[np.linalg.norm(prey_pos - pred_pos) for pred_pos in predator_positions] for prey_pos in prey_positions])
        min_distance = distances.min()

        # Calculate rewards based on distance changes
        prey_rewards = np.sum(distances - self.prev_distances, axis=1) / self.num_predators
        predator_rewards = np.sum(self.prev_distances - distances, axis=0) / self.num_prey

        for i in range(self.num_prey):
            rewards[self.agents[i]] = prey_rewards[i]

        for i in range(self.num_predators):
            rewards[self.agents[self.num_prey + i]] = predator_rewards[i]

        # Update previous distances
        self.prev_distances = distances

        # Implement termination condition: if min_distance < 1 or max steps reached
        if min_distance < 1:
            terminations = {agent: True for agent in self.agents}
        if self.current_step >= self.max_steps:
            truncations = {agent: True for agent in self.agents}

        observations = self.get_observations()
        return observations, rewards, terminations, truncations, infos

    def render(self, mode="human"):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))

        self.screen.fill((255, 255, 255))

        # Draw prey
        for i in range(self.num_prey):
            pygame.draw.circle(self.screen, (0, 255, 0), (self.agent_positions[i] * self.scale).astype(int), self.scale)

        # Draw predators
        for i in range(self.num_predators):
            pygame.draw.circle(self.screen, (255, 0, 0), (self.agent_positions[self.num_prey + i] * self.scale).astype(int), self.scale)

        pygame.display.flip()

    def get_observations(self):
        observations = {}
        prey_positions = self.agent_positions[:self.num_prey]
        predator_positions = self.agent_positions[self.num_prey:]

        for i, agent in enumerate(self.agents):
            if "prey" in agent:
                # Get x and y distances to each predator
                distances = []
                for pred_pos in predator_positions:
                    x_dist = prey_positions[i][0] - pred_pos[0]
                    y_dist = prey_positions[i][1] - pred_pos[1]
                    distances.extend([x_dist, y_dist])
                # Normalize the distances by grid size
                observations[agent] = np.array(distances) / self.grid_size

            elif "predator" in agent:
                # Get x and y distances to each prey
                distances = []
                for prey_pos in prey_positions:
                    x_dist = predator_positions[i - self.num_prey][0] - prey_pos[0]
                    y_dist = predator_positions[i - self.num_prey][1] - prey_pos[1]
                    distances.extend([x_dist, y_dist])
                # Normalize the distances by grid size
                observations[agent] = np.array(distances) / self.grid_size

        return observations

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
