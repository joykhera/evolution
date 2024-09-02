import numpy as np
from pettingzoo.utils import ParallelEnv
from gymnasium import spaces
import pygame
from agent import Agent


class TagEnv(ParallelEnv):
    metadata = {"render.modes": ["human"], "name": "custom_tag_v0"}

    def __init__(
        self,
        prey_count=1,
        prey_speed=1,
        prey_view_size=10,
        prey_size=1,
        prey_color=(0, 255, 0),  # Green
        prey_kill_reward=-10,
        prey_alive_reward=0.1,
        prey_reward_sqr_reward=1,
        predator_count=1,
        predator_speed=1,
        predator_view_size=10,
        predator_size=1,
        predator_color=(255, 0, 0),  # Red
        predator_kill_reward=10,
        map_size=50,
        max_steps=400,
        screen_size=600,
        render_mode=None,
        fps=60,
    ):
        # Prey-specific initialization
        self.prey_count = prey_count
        self.prey_speed = prey_speed
        self.prey_view_size = prey_view_size
        self.prey_size = prey_size
        self.prey_color = prey_color
        self.prey_kill_reward = prey_kill_reward
        self.prey_alive_reward = prey_alive_reward
        self.prey_reward_sqr_reward = prey_reward_sqr_reward

        # Predator-specific initialization
        self.predator_count = predator_count
        self.predator_speed = predator_speed
        self.predator_view_size = predator_view_size
        self.predator_size = predator_size
        self.predator_color = predator_color
        self.predator_kill_reward = predator_kill_reward

        # General environment initialization
        self.map_size = map_size
        self.max_steps = max_steps
        self.screen_size = screen_size
        self.render_mode = render_mode
        self.fps = fps
        self.clock = None

        # Initialize other variables and environment components
        self.agents = {}
        self.possible_agents = [f"prey_{i}" for i in range(prey_count)] + [f"predator_{i}" for i in range(predator_count)]
        self.current_step = 0
        self.scale = self.screen_size // self.map_size
        self.canvas = pygame.Surface((self.map_size, self.map_size))

        self.reward_square_size = self.map_size // 2
        self.reward_square_start = (self.map_size - self.reward_square_size) // 2
        self.reward_square_end = self.reward_square_start + self.reward_square_size

        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
            self.scaled_canvas = pygame.Surface((self.screen_size, self.screen_size))
            pygame.display.set_caption("Tag Environment")
            pygame.font.init()
            self.font = pygame.font.SysFont(None, 24)
            self.clock = pygame.time.Clock()
            pygame.font.init()

        self.observation_spaces = {}
        for agent in self.possible_agents:
            if "prey" in agent:
                self.observation_spaces[agent] = spaces.Box(low=0, high=1, shape=(self.prey_view_size, self.prey_view_size, 3), dtype=np.uint8)
            elif "predator" in agent:
                self.observation_spaces[agent] = spaces.Box(low=0, high=1, shape=(self.predator_view_size, self.predator_view_size, 3), dtype=np.uint8)
        self.action_spaces = {agent: spaces.Discrete(5) for agent in self.possible_agents}

    def _init_agents(self):
        for i in range(self.prey_count):
            position = np.random.randint(0, self.map_size, size=2)
            self.agents[f"prey_{i}"] = Agent(
                agent_type="prey",
                position=position,
                size=self.prey_size,
                speed=self.prey_speed,
                color=self.prey_color,
                map_size=self.map_size,
                view_size=self.prey_view_size,
                scale=self.scale,
            )
        for i in range(self.predator_count):
            # if self.prey_count <= self.predator_count:
            #     prey_pos = self.agents[f"prey_{i % self.prey_count}"].position
            # else:
            #     prey_pos = self.agents[f"prey_{np.random.randint(self.prey_count)}"].position
            # half_view_size = self.predator_view_size // 2
            # position = [
            #     np.random.randint(max(0, prey_pos[0] - half_view_size), min(self.map_size, prey_pos[0] + half_view_size)),
            #     np.random.randint(max(0, prey_pos[1] - half_view_size), min(self.map_size, prey_pos[1] + half_view_size)),
            # ]
            position = np.random.randint(0, self.map_size, size=2)
            self.agents[f"predator_{i}"] = Agent(
                agent_type="predator",
                position=position,
                size=self.predator_size,
                speed=self.predator_speed,
                color=self.predator_color,
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
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        for agent_id, action in actions.items():
            self.agents[agent_id].move(action)
            # print(f"{agent_id} moved to {self.agents[agent_id].position}")

        rewards = self.compute_rewards(terminations)

        # Remove terminated prey
        prey_left = False
        for agent_id, terminated in list(terminations.items()):
            if terminated:
                del self.agents[agent_id]
            elif "prey" in agent_id:
                prey_left = True

        # Check if all prey are caught (no prey left)
        if not prey_left:
            for agent_id in self.agents:
                terminations[agent_id] = True  # End episode for all agents

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
        # self.canvas.fill((255, 255, 255))
        self.canvas.fill(1)

        pygame.draw.rect(
            self.canvas,
            (0, 0, 255),
            (
                self.reward_square_start,
                self.reward_square_start,
                self.reward_square_size,
                self.reward_square_size,
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
                self.reward_square_start * self.scale,
                self.reward_square_start * self.scale,
                self.reward_square_size * self.scale,
                self.reward_square_size * self.scale,
            ),
        )

        for agent in self.agents.values():
            # agent.draw(self.scaled_canvas, render_mode="human", draw_grid=True)
            agent.draw(self.scaled_canvas, render_mode="human")

        predator_score_text = self.font.render(f"Predator score: {round(self.predator_score)}", True, (0, 0, 0))
        prey_score_text = self.font.render(f"Prey score: {round(self.prey_score)}", True, (0, 0, 0))
        prey_eaten_text = self.font.render(f"Prey eaten: {self.prey_eaten}", True, (0, 0, 0))
        self.screen.blit(self.scaled_canvas, (0, 0))
        self.screen.blit(predator_score_text, (5, 10))
        self.screen.blit(prey_score_text, (5, 25))
        self.screen.blit(prey_eaten_text, (5, 40))
        pygame.display.flip()
        self.clock.tick(self.fps)

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

                if min_distance <= 0.5:
                    return 0
        return min_distance

    def compute_rewards(self, terminations):
        rewards = {}
        alive_preys = []
        alive_predators = []

        # Collect positions of alive prey and predators in lists for direct access
        for agent_id, agent in self.agents.items():
            if "prey" in agent_id:
                alive_preys.append((agent_id, agent.position))
            elif "predator" in agent_id:
                alive_predators.append((agent_id, agent.position))
        # print(alive_predators, alive_preys)
        # Prey rewards
        for prey_id, prey_pos in alive_preys:
            prey_half_grid = self.prey_view_size // 2  # Use prey's view size
            prey_reward = 0

            # Calculate the minimum distance to any predator
            min_distance_to_predator = float('inf')
            for _, predator_pos in alive_predators:
                distance = self.calculate_min_distance(prey_pos, [predator_pos], prey_half_grid)
                min_distance_to_predator = min(min_distance_to_predator, distance)

                if min_distance_to_predator == 0:  # Stop further checks if a predator catches the prey
                    break

            # Determine prey reward based on proximity to predators
            if min_distance_to_predator == 0:  # Predator touches the prey
                prey_reward = self.prey_kill_reward  # Larger negative reward for being caught
                terminations[prey_id] = True  # Mark prey as terminated
                self.prey_eaten += 1
            elif min_distance_to_predator < float('inf'):
                prey_reward -= prey_half_grid / min_distance_to_predator
            elif self.reward_square_start <= prey_pos[0] <= self.reward_square_end and self.reward_square_start <= prey_pos[1] <= self.reward_square_end:
                prey_reward += self.prey_reward_sqr_reward  # Reward for being on the blue square
            else:
                prey_reward += self.prey_alive_reward  # Small positive reward for staying alive

            rewards[prey_id] = prey_reward
            self.prey_score += prey_reward

        # Predator rewards
        for predator_id, predator_pos in alive_predators:
            predator_half_grid = self.predator_view_size // 2  # Use predator's view size
            predator_reward = 0

            # Calculate the minimum distance to any prey
            min_distance_to_prey = float('inf')
            for _, prey_pos in alive_preys:
                distance = self.calculate_min_distance(predator_pos, [prey_pos], predator_half_grid)
                min_distance_to_prey = min(min_distance_to_prey, distance)

                if min_distance_to_prey == 0:  # Stop further checks if a predator catches the prey
                    break

            # Determine predator reward based on proximity to preys
            if min_distance_to_prey == 0:  # Predator touches the prey
                predator_reward = self.predator_kill_reward  # Larger reward for catching the prey
            elif min_distance_to_prey < float('inf'):
                predator_reward += predator_half_grid / min_distance_to_prey

            rewards[predator_id] = predator_reward
            self.predator_score += predator_reward

        return rewards

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
