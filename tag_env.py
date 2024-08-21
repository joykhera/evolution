import numpy as np
from pettingzoo.utils import ParallelEnv
from gymnasium import spaces
import pygame
from tag_agent import Agent


# Color Encoding:
# 0 for black
# 1 for white
# 2 for green
# 3 for red
# 4 for blue


class TagEnv(ParallelEnv):
    metadata = {"render.modes": ["human"], "name": "custom_tag_v0"}

    def __init__(
        self,
        num_prey=1,
        num_predators=2,
        map_size=50,
        max_steps=400,
        screen_size=600,
        predator_speed=1,
        prey_speed=1,
        prey_view_size=10,
        predator_view_size=10,
        render_mode=None,
    ):
        self.num_prey = num_prey
        self.num_predators = num_predators
        self.map_size = map_size
        self.max_steps = max_steps
        self.screen_size = screen_size
        self.predator_speed = predator_speed
        self.prey_speed = prey_speed
        self.prey_view_size = prey_view_size
        self.predator_view_size = predator_view_size
        self.render_mode = render_mode
        self.agents = {}
        self.possible_agents = [f"prey_{i}" for i in range(num_prey)] + [f"predator_{i}" for i in range(num_predators)]
        self.current_step = 0
        self.scale = self.screen_size // self.map_size
        self.canvas = pygame.Surface((self.map_size, self.map_size))

        self.blue_square_size = self.map_size // 2
        # self.blue_square_size = 0
        self.blue_square_start = (self.map_size - self.blue_square_size) // 2
        self.blue_square_end = self.blue_square_start + self.blue_square_size

        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
            self.scaled_canvas = pygame.Surface((self.screen_size, self.screen_size))
            pygame.font.init()
            self.font = pygame.font.SysFont(None, 24)

        # self.observation_spaces = {agent: spaces.Box(low=0, high=1, shape=(prey_view_size, prey_view_size), dtype=np.uint8) for agent in self.possible_agents}
        self.observation_spaces = {}
        for agent in self.possible_agents:
            if "prey" in agent:
                # self.observation_spaces[agent] = spaces.Box(low=0, high=1, shape=(prey_view_size, prey_view_size), dtype=np.uint8)
                self.observation_spaces[agent] = spaces.Box(low=0, high=1, shape=(prey_view_size, prey_view_size, 3), dtype=np.uint8)
            elif "predator" in agent:
                # self.observation_spaces[agent] = spaces.Box(low=0, high=1, shape=(predator_view_size, predator_view_size), dtype=np.uint8)
                self.observation_spaces[agent] = spaces.Box(low=0, high=1, shape=(predator_view_size, predator_view_size, 3), dtype=np.uint8)
        self.action_spaces = {agent: spaces.Discrete(5) for agent in self.possible_agents}

    def _init_agents(self):
        for i in range(self.num_prey):
            position = np.random.rand(2) * self.map_size
            self.agents[f"prey_{i}"] = Agent(
                agent_type="prey",
                position=position,
                size=1,
                speed=self.prey_speed,
                color=(0, 255, 0),
                color_encoding=2,
                map_size=self.map_size,
                view_size=self.prey_view_size,
                scale=self.scale,
            )
        for i in range(self.num_predators):
            if self.num_prey <= self.num_predators:
                prey_pos = self.agents[f"prey_{i % self.num_prey}"].position
            else:
                prey_pos = self.agents[f"prey_{np.random.randint(self.num_prey)}"].position
            half_view_size = self.predator_view_size // 2
            position = [
                np.random.randint(max(0, prey_pos[0] - half_view_size), min(self.map_size, prey_pos[0] + half_view_size)),
                np.random.randint(max(0, prey_pos[1] - half_view_size), min(self.map_size, prey_pos[1] + half_view_size)),
            ]
            self.agents[f"predator_{i}"] = Agent(
                agent_type="predator",
                position=position,
                size=1,
                speed=self.predator_speed,
                color=(255, 0, 0),
                color_encoding=3,
                map_size=self.map_size,
                view_size=self.predator_view_size,
                scale=self.scale,
            )

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
        # print('aaa', observations)

        return observations, rewards, terminations, truncations, infos

    def render(self):
        self.canvas.fill(1)

        pygame.draw.rect(
            self.canvas,
            4,
            (
                self.blue_square_start,
                self.blue_square_start,
                self.blue_square_size,
                self.blue_square_size,
            ),
        )

        for agent in self.agents.values():
            agent.draw(self.canvas)

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
            agent.draw(self.scaled_canvas, render_mode="human", draw_grid=True)

        predator_score_text = self.font.render(f"Predator score: {round(self.predator_score)}", True, (0, 0, 0))
        prey_score_text = self.font.render(f"Prey score: {round(self.prey_score)}", True, (0, 0, 0))
        prey_eaten_text = self.font.render(f"Prey eaten: {self.prey_eaten}", True, (0, 0, 0))
        self.screen.blit(self.scaled_canvas, (0, 0))
        self.screen.blit(predator_score_text, (5, 10))
        self.screen.blit(prey_score_text, (5, 25))
        self.screen.blit(prey_eaten_text, (5, 40))
        pygame.display.flip()

    def get_observations(self):
        observations = {}
        for agent_id, agent in self.agents.items():
            observations[agent_id] = agent.get_observation(self.canvas)
        return observations

    def calculate_min_distance(self, agent_pos, other_agents_positions, half_view_size):
        min_distance = float('inf')

        for other_pos in other_agents_positions:
            x_distance = abs(agent_pos[0] - other_pos[0])
            y_distance = abs(agent_pos[1] - other_pos[1])

            if x_distance < half_view_size and y_distance < half_view_size:
                distance = np.sqrt(x_distance**2 + y_distance**2)
                min_distance = min(min_distance, distance)

                if min_distance < 0.5:
                    self.prey_eaten += 1
                    return 0
        return min_distance

    def compute_rewards(self):
        rewards = {}
        prey_positions = [self.agents[f"prey_{i}"].position for i in range(self.num_prey)]
        predator_positions = [self.agents[f"predator_{i}"].position for i in range(self.num_predators)]

        # Prey rewards
        for i in range(self.num_prey):
            prey_pos = prey_positions[i]
            prey_half_grid = self.prey_view_size // 2  # Use prey's view size

            min_distance_to_predator = self.calculate_min_distance(prey_pos, predator_positions, prey_half_grid)
            prey_reward = 0
            # print("min_distance_to_predator", min_distance_to_predator)
            if min_distance_to_predator == 0:  # Predator touches the prey
                prey_reward = -10  # Larger penalty for being caught
                self.prey_eaten += 1
            elif min_distance_to_predator < float('inf'):
                prey_reward -= prey_half_grid / min_distance_to_predator
                # prey_reward += min_distance_to_predator
            elif self.blue_square_start <= prey_pos[0] <= self.blue_square_end and self.blue_square_start <= prey_pos[1] <= self.blue_square_end:
                prey_reward += 1  # Reward for being on the blue square
            else:
                prey_reward += 0.1  # Small positive reward for staying alive

            rewards[f"prey_{i}"] = prey_reward
            self.prey_score += prey_reward

        # Predator rewards
        for i in range(self.num_predators):
            predator_pos = predator_positions[i]
            predator_half_grid = self.predator_view_size // 2  # Use predator's view size

            min_distance_to_prey = self.calculate_min_distance(predator_pos, prey_positions, predator_half_grid)
            predator_reward = 0

            if min_distance_to_prey == 0:  # Predator touches the prey
                predator_reward = 10  # Larger reward for catching the prey
            elif min_distance_to_prey < float('inf'):
                # Positive reward proportional to how close the predator is to the prey
                predator_reward += predator_half_grid / min_distance_to_prey
                # predator_reward = predator_half_grid - min_distance_to_prey

            rewards[f"predator_{i}"] = predator_reward
            self.predator_score += predator_reward

        return rewards

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
