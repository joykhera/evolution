import numpy as np
from pettingzoo.utils import ParallelEnv
from gymnasium import spaces
import pygame
from tag_agent import Agent


class TagEnv(ParallelEnv):
    metadata = {"render.modes": ["human"], "name": "custom_tag_v0"}

    def __init__(self, num_prey=1, num_predators=2, map_size=50, max_steps=400, screen_size=600, predator_speed=1, prey_speed=1, grid_size=10, render_mode=None):
        self.num_prey = num_prey
        self.num_predators = num_predators
        self.map_size = map_size
        self.max_steps = max_steps
        self.screen_size = screen_size
        self.predator_speed = predator_speed
        self.prey_speed = prey_speed
        self.grid_size = grid_size
        self.render_mode = render_mode
        self.agents = {}
        self.possible_agents = [f"prey_{i}" for i in range(num_prey)] + [f"predator_{i}" for i in range(num_predators)]
        self.current_step = 0
        self.scale = self.screen_size // self.map_size
        self.canvas = pygame.Surface((self.map_size, self.map_size))

        self.blue_square_size = self.map_size // 2
        self.blue_square_start = (self.map_size - self.blue_square_size) // 2
        self.blue_square_end = self.blue_square_start + self.blue_square_size

        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
            self.scaled_canvas = pygame.Surface((self.screen_size, self.screen_size))
            pygame.font.init()
            self.font = pygame.font.SysFont(None, 24)

        self.observation_spaces = {agent: spaces.Box(low=0, high=4, shape=(grid_size, grid_size), dtype=np.uint8) for agent in self.possible_agents}
        self.action_spaces = {agent: spaces.Discrete(5) for agent in self.possible_agents}

    def _init_agents(self):
        for i in range(self.num_prey):
            position = np.random.rand(2) * self.map_size
            self.agents[f"prey_{i}"] = Agent(position, size=1, speed=self.prey_speed, color=(0, 255, 0), map_size=self.map_size, grid_size=self.grid_size)
        for i in range(self.num_predators):
            position = np.random.rand(2) * self.map_size
            self.agents[f"predator_{i}"] = Agent(position, size=1, speed=self.predator_speed, color=(255, 0, 0), map_size=self.map_size, grid_size=self.grid_size)

    def reset(self, seed=None, return_info=False, options=None):
        self.current_step = 0
        self._init_agents()
        self.predator_score = 0
        self.prey_score = 0
        self.prey_eaten = 0

        observations = self.get_observations()
        return observations, {}

    def step(self, actions):
        self.current_step += 1
        terminations = {agent: False for agent in self.possible_agents}
        truncations = {agent: False for agent in self.possible_agents}
        infos = {agent: {} for agent in self.possible_agents}

        for agent_id, action in actions.items():
            self.agents[agent_id].move(action)
            # print(f"{agent_id} moved to {self.agents[agent_id].position}")

        rewards = self.compute_rewards()

        if self.current_step >= self.max_steps:
            for agent in self.possible_agents:
                truncations[agent] = True

        self.render()
        if self.render_mode == "human":
            self.render_human()

        observations = self.get_observations()

        return observations, rewards, terminations, truncations, infos

    def render(self):
        self.canvas.fill((255, 255, 255))

        pygame.draw.rect(
            self.canvas,
            (0, 0, 255),
            (
                self.blue_square_start,
                self.blue_square_start,
                self.blue_square_size,
                self.blue_square_size,
            ),
        )

        for agent in self.agents.values():
            agent.draw(self.canvas, 1)

    def render_human(self):
        self.scaled_canvas.fill((255, 255, 255))
        pygame.draw.rect(
            self.scaled_canvas,
            (0, 0, 255),
            (
                self.blue_square_start * self.scale,
                self.blue_square_start * self.scale,
                self.blue_square_size * self.scale,
                self.blue_square_size * self.scale,
            ),
        )

        for agent in self.agents.values():
            agent.draw(self.scaled_canvas, self.scale, draw_grid=True)

        predator_score_text = self.font.render(f"Predator score: {self.predator_score}", True, (0, 0, 0))
        prey_score_text = self.font.render(f"Prey score: {self.prey_score}", True, (0, 0, 0))
        prey_eaten_text = self.font.render(f"Prey eaten: {self.prey_eaten}", True, (0, 0, 0))
        self.screen.blit(self.scaled_canvas, (0, 0))
        self.screen.blit(predator_score_text, (5, 10))
        self.screen.blit(prey_score_text, (5, 25))
        self.screen.blit(prey_eaten_text, (5, 40))
        pygame.display.flip()

    def get_observations(self):
        observations = {}
        for agent_id, agent in self.agents.items():
            observation = agent.get_observation(self.canvas)
            normalized_observation = observation / 255
            observations[agent_id] = normalized_observation
        return observations

    def compute_rewards(self):
        rewards = {}
        prey_positions = [self.agents[f"prey_{i}"].position for i in range(self.num_prey)]
        predator_positions = [self.agents[f"predator_{i}"].position for i in range(self.num_predators)]
        half_grid = self.grid_size // 2

        # Prey rewards
        for i in range(self.num_prey):
            x, y = prey_positions[i]
            prey_in_danger = False
            prey_reward = 0

            for predator_pos in predator_positions:
                x_distance = abs(x - predator_pos[0])
                y_distance = abs(y - predator_pos[1])

                if x_distance < half_grid and y_distance < half_grid:
                    # Predator is within the grid of the prey
                    prey_reward = -2  # Penalty if prey and predator see each other
                    prey_in_danger = True

                    if x_distance == 0 and y_distance == 0:  # Predator touches the prey
                        prey_reward = -10  # Larger penalty for being caught
                        self.prey_eaten += 1
                        break

            if not prey_in_danger:
                if self.blue_square_start <= x <= self.blue_square_end and self.blue_square_start <= y <= self.blue_square_end:
                    prey_reward = 1  # Reward for being on the blue square
                else:
                    prey_reward = 0.1

            rewards[f"prey_{i}"] = prey_reward
            self.prey_score += prey_reward

        # Predator rewards
        for i in range(self.num_predators):
            predator_pos = predator_positions[i]
            predator_reward = 0
            predator_has_prey_in_view = False

            for prey_pos in prey_positions:
                x_distance = abs(predator_pos[0] - prey_pos[0])
                y_distance = abs(predator_pos[1] - prey_pos[1])

                if x_distance < half_grid and y_distance < half_grid:
                    predator_reward = 1  # Reward if predator sees the prey
                    predator_has_prey_in_view = True

                    if x_distance == 0 and y_distance == 0:  # Predator touches the prey
                        predator_reward = 10  # Larger reward for catching the prey
                        break

            if not predator_has_prey_in_view:
                predator_reward = 0  # No reward if no prey is in the predator's view

            rewards[f"predator_{i}"] = predator_reward
            self.predator_score += predator_reward

        return rewards

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
