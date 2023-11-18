import pygame
import random
import numpy as np
from game.food import Food
from game.player import Player
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Constants

SIZE = 50
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
FOOD_COUNT = 100
FPS = 60
FOOD_SIZE = SIZE / 50
PLAYER_SIZE = SIZE / 50
PLAYER_SIZE_INCREASE = PLAYER_SIZE / 10
PLAYER_SPEED = SIZE / 100
GRID_SIZE = 10
EPISODE_STEPS = 100
SCALE = 5
WINDOW_SIZE = SIZE + GRID_SIZE

# plt.ion()  # Interactive mode on
# print("Interactive mode on")
# # fig, ax = plt.subplots()  # Create a new figure and set of subplots
# fig1, ax1 = plt.subplots()  # Create a new figure and set of subplots
# image = ax1.imshow(np.zeros((100, 100)), "BrBG")
# plt.show()  # Show the figure


class Game:
    """Main game class that handles game logic."""

    def __init__(self, num_agents=50, human_player=True):
        pygame.init()
        pygame.display.init()
        self.human_player = human_player
        self.screen = pygame.display.set_mode(
            (WINDOW_SIZE * SCALE, WINDOW_SIZE * SCALE)
        )
        pygame.display.set_caption("Evolution Environment")
        self.clock = pygame.time.Clock()
        self.num_agents = num_agents
        self.players = []
        self.foods = []
        self.running = True
        self.steps = 0
        self.window = pygame.display.set_mode(
            (WINDOW_SIZE * SCALE, WINDOW_SIZE * SCALE)
        )
        self.canvas = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE))
        self.prev_step_time = time.perf_counter()

    def handle_events(self):
        """Handles game events, such as input and quitting."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def get_action_vector(self, action):
        """Converts an action number into a direction vector."""
        actions = [
            pygame.Vector2(0, 0),  # None
            pygame.Vector2(1, 0),  # Right
            pygame.Vector2(-1, 0),  # Left
            pygame.Vector2(0, -1),  # Up
            pygame.Vector2(0, 1),  # Down
        ]
        return actions[action] if action is not None else pygame.Vector2(0, 0)

    def step(self, actions):
        """Executes a step for each agent in the environment."""
        observations = []
        rewards = []

        for i, action in enumerate(actions):
            if self.human_player:
                direction_vector = self.get_input()
            else:
                max_action = self.output_to_action(action)
                direction_vector = self.get_action_vector(max_action)

            self.players[i].move(direction_vector)
            observation = self.get_observation(player_idx=i)
            reward = self.get_reward(i)

            # if i == 0:
            #     image.set_data(observation)  # Update the image with the observation
            #     ax1.clear()  # Clear previous annotations
            #     ax1.imshow(observation)  # Show the updated observation
            #     ax1.text(
            #         0.15,
            #         0.95,
            #         f"Reward: {reward}",
            #         color="white",
            #         ha="center",
            #         va="center",
            #         transform=ax1.transAxes,
            #         bbox=dict(facecolor="black", alpha=0.5),
            #     )
            #     plt.pause(0.001)  # Pause to update the plot

            observations.append(observation)
            rewards.append(reward)

        done = self.is_done()
        self.steps += 1
        cur_time = time.perf_counter()
        print("Step:", self.steps, " Step time: ", cur_time - self.prev_step_time)
        self.prev_step_time = cur_time
        return observations, rewards, done

    def get_input(self):
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

    def check_collisions(self, player_idx):
        """Checks for collisions between player and food."""
        for food in self.foods[:]:
            if (
                self.players[player_idx].position.distance_to(food.position)
                < self.players[player_idx].size
            ):
                # self.foods.remove(food)
                # print("Food eaten by player ", player_idx)
                return True

        return False

    def is_on_black(self, player_idx):
        """Returns True if the player is on a black pixel, otherwise False."""
        player = self.players[player_idx]
        # Convert player's position to integer for pixel access
        player_pos = (int(player.position.x), int(player.position.y))

        # Get the pixel array from the canvas
        pixels = pygame.surfarray.array3d(self.canvas)

        # Check the color at the player's position
        # Since the pixel array is transposed, access it with y first, x second
        if tuple(pixels[player_pos[1], player_pos[0]]) == BLACK:
            return True
        else:
            return False

    def apply_function_to_all_agents(self, function):
        """Applies a function to all agents and return"""
        return [function(i) for i in range(len(self.players))]

    def cast_ray(self, agent_position, direction, foods, max_distance, num_rays=8):
        """
        Cast a ray in the given direction to find the distance to the nearest food.

        :param agent_position: The position of the agent casting the ray
        :param direction: The direction in which to cast the ray
        :param foods: The list of food positions
        :param max_distance: The maximum distance a ray can travel (e.g., the diagonal of the environment)
        :param num_rays: The number of rays to cast (equally spaced in 360 degrees)
        :return: The distance to the nearest food in the direction of the ray, normalized by max_distance
        """
        ray_distances = np.full(num_rays, max_distance)
        angle_between_rays = 360 / num_rays

        for i, ray_angle in enumerate(np.arange(0, 360, angle_between_rays)):
            ray_vector = pygame.Vector2(1, 0).rotate(
                -ray_angle
            )  # Pygame's y-axis is flipped
            nearest_distance = max_distance

            for food in foods:
                food_vector = pygame.Vector2(food.position) - agent_position
                food_distance = agent_position.distance_to(food.position)
                angle_to_food = ray_vector.angle_to(food_vector)

                # Check if the food is within the angle threshold to be considered "in the direction" of the ray
                if (
                    abs(angle_to_food) < angle_between_rays / 2
                    and food_distance < nearest_distance
                ):
                    nearest_distance = food_distance

            # Normalize the distance
            ray_distances[i] = nearest_distance / max_distance

        return ray_distances

    def get_observation(self, player_idx):
        """
        Get the observation for a player using ray casting.

        :param player_idx: Index of the player for which to get the observation
        :return: An array of normalized distances in the direction of each ray
        """
        player_position = self.players[player_idx].position
        max_distance = int(
            np.sqrt(self.screen.get_width() ** 2 + self.screen.get_height() ** 2)
        )

        # Get the distances to the nearest food in the direction of each ray
        # observation = self.cast_ray(
        #     agent_position=player_position,
        #     direction=pygame.Vector2(1, 0),
        #     foods=self.foods,
        #     max_distance=max_distance,
        # )
        observation = self.cast_ray(
            player_position, pygame.Vector2(1, 0), self.foods, max_distance
        )
        print(observation)
        return observation

    def draw_rays(self, surface, agent_position, foods, max_distance, num_rays=8):
        angle_between_rays = 360 / num_rays
        for ray_angle in np.arange(0, 360, angle_between_rays):
            ray_vector = pygame.Vector2(1, 0).rotate(-ray_angle)
            nearest_distance = max_distance
            nearest_food = None

            # Find the nearest food in the direction of the ray
            for food in foods:
                food_vector = pygame.Vector2(food.position) - agent_position
                food_distance = food_vector.length()
                angle_to_food = ray_vector.angle_to(food_vector)

                if (
                    abs(angle_to_food) < angle_between_rays / 2
                    and food_distance < nearest_distance
                ):
                    nearest_distance = food_distance
                    nearest_food = food_vector + agent_position

            # Draw the ray up to the nearest food or max distance
            if nearest_food:
                end_point = nearest_food
            else:
                end_point = agent_position + ray_vector.normalize() * max_distance

            pygame.draw.line(surface, (255, 255, 0), agent_position, end_point, 1)

    def _clamp_vector(self, vector, min_vector, max_vector):
        return pygame.Vector2(
            max(min_vector.x, min(vector.x, max_vector.x)),
            max(min_vector.y, min(vector.y, max_vector.y)),
        )

    def get_reward(self, player_idx):
        """Returns the reward after an action."""
        return int(self.check_collisions(player_idx))
        # return (
        #     int(self.check_collisions(player_idx))
        #     - int(self.is_on_black(player_idx)) / 1
        # )
        # return int(not self.is_on_black(player_idx))
        # return self.players[player_idx].position.x > WINDOW_SIZE / 1.2
        # player = self.players[player_idx]
        # player_rect = pygame.Rect(
        #     player.position.x, player.position.y, PLAYER_SIZE, PLAYER_SIZE
        # )

        # # Check if the player is within the green square
        # if self.green_rect.colliderect(player_rect):
        #     return 1  # Positive reward
        # else:
        #     return -1  # No reward

    def is_done(self):
        """Checks if the game is finished."""
        return self.steps >= EPISODE_STEPS
        # return self.steps >= EPISODE_STEPS or self.foods == []

    def reset(self):
        """Resets the game to an initial state."""
        # self.foods = [
        #     Food(
        #         position=(
        #             random.randint(GRID_SIZE // 2, SIZE),
        #             random.randint(GRID_SIZE // 2, SIZE),
        #         ),
        #         size=FOOD_SIZE,
        #         color=GREEN,
        #     )
        #     for _ in range(FOOD_COUNT)
        # ]
        food_per_row = int(np.sqrt(FOOD_COUNT))
        food_per_col = FOOD_COUNT // food_per_row

        # Calculate spacing between food items
        x_spacing = (SIZE) // food_per_row
        y_spacing = (SIZE) // food_per_col

        # Create a grid of food positions
        self.foods = []
        for i in range(food_per_row):
            for j in range(food_per_col):
                x_position = GRID_SIZE + i * x_spacing
                y_position = GRID_SIZE + j * y_spacing
                self.foods.append(
                    Food(
                        position=(x_position, y_position),
                        size=FOOD_SIZE,
                        color=GREEN,
                    )
                )
        self.players = [
            Player(
                position=(
                    random.randint(GRID_SIZE // 2, SIZE),
                    random.randint(GRID_SIZE // 2, SIZE),
                ),
                map_size=WINDOW_SIZE,
                size=PLAYER_SIZE,
                speed=PLAYER_SPEED,
                color=RED,
            )
            for _ in range(self.num_agents)
        ]
        # self.players[0].color = (0, 0, 255)
        observations = [
            self.get_observation(player_idx=i) for i in range(self.num_agents)
        ] * self.num_agents
        self.steps = 0
        return observations

    def render(self, mode="human", scale=1):
        # Clear the canvas with a black color
        self.canvas.fill(BLACK)

        pygame.draw.rect(
            self.canvas,
            WHITE,
            pygame.Rect(GRID_SIZE // 2, GRID_SIZE // 2, SIZE, SIZE),
        )
        for food in self.foods:
            food.draw(self.canvas, 1)
        for player in self.players:
            player.draw(self.canvas, 1)

        max_distance = int(
            np.sqrt(self.screen.get_width() ** 2 + self.screen.get_height() ** 2)
        )
        # self.draw_rays(self.canvas, self.players[0].position, self.foods, max_distance)

        scaled_canvas = pygame.Surface((WINDOW_SIZE * scale, WINDOW_SIZE * scale))
        scaled_canvas.blit(
            pygame.transform.scale(
                self.canvas, (WINDOW_SIZE * scale, WINDOW_SIZE * scale)
            ),
            scaled_canvas.get_rect(),
        )

        if mode == "human":
            self.window.blit(scaled_canvas, scaled_canvas.get_rect())
            pygame.event.pump()
            self.clock.tick(FPS)
            pygame.display.flip()

    def close(self):
        """Closes the Pygame window."""
        pygame.quit()

    def test_ai(self, nets, num_steps=1000):
        """Test the AI's performance in the game for a certain number of steps."""
        self.reset()
        pygame.display.set_caption("NEAT Test AI")
        for _ in range(num_steps):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()
            self.screen.fill(WHITE)

            # observation = self.get_observation()
            observations = [
                self.get_observation(player_idx=i) for i in range(self.num_agents)
            ]

            actions = [
                net.activate(obs.flatten()) for net, obs in zip(nets, observations)
            ]

            self.step(actions)
            self.render(scale=SCALE)
            self.clock.tick(FPS)

            if self.is_done():
                break

    def output_to_action(self, output):
        return output.index(max(output))


def run_game_with_human():
    game = Game(human_player=True)
    game.run()


if __name__ == "__main__":
    run_game_with_human()
