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
FOOD_COUNT = 50
FPS = 60
FOOD_SIZE = SIZE / 50
PLAYER_SIZE = SIZE / 50
PLAYER_SIZE_INCREASE = PLAYER_SIZE / 10
PLAYER_SPEED = SIZE / 100
GRID_SIZE = 10
EPISODE_STEPS = 100
SCALE = 10
WINDOW_SIZE = SIZE + GRID_SIZE

# plt.ion()  # Interactive mode on
# fig, ax = plt.subplots()  # Create a new figure and set of subplots
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
            for _ in range(num_agents)
        ]
        self.foods = [
            Food(
                position=(
                    random.randint(GRID_SIZE // 2, SIZE),
                    random.randint(GRID_SIZE // 2, SIZE),
                ),
                size=FOOD_SIZE,
                color=GREEN,
            )
            for _ in range(FOOD_COUNT)
        ]
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
            #     image.set_data(observation)
            #     plt.pause(0.001)

            observations.append(observation)
            rewards.append(reward)

        done = self.is_done()
        self.steps += 1
        cur_time = time.perf_counter()
        # print("Step:", self.steps, " Step time: ", cur_time - self.prev_step_time)
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
                self.foods.remove(food)
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

    # def get_observation(self):
    #     """Returns the entire screen as a flattened numpy array of pixel values for NEAT."""
    #     canvas_arr = np.transpose(
    #         np.array(pygame.surfarray.pixels3d(self.canvas)), axes=(1, 0, 2)
    #     )
    #     norm_observation = canvas_arr / 255
    #     return norm_observation

    def get_observation(self, player_idx):
        player_center = self.players[player_idx].position
        top_left = player_center - pygame.Vector2(GRID_SIZE // 2, GRID_SIZE // 2)
        bottom_right = player_center + pygame.Vector2(GRID_SIZE // 2, GRID_SIZE // 2)
        # print("before clamp", player_center, top_left, bottom_right)
        # Make sure we don't go out of bounds
        top_left = self._clamp_vector(
            top_left,
            pygame.Vector2(0, 0),
            pygame.Vector2(WINDOW_SIZE, WINDOW_SIZE) + pygame.Vector2(1, 1),
        )
        bottom_right = self._clamp_vector(
            bottom_right,
            pygame.Vector2(0, 0),
            pygame.Vector2(WINDOW_SIZE, WINDOW_SIZE) - pygame.Vector2(1, 1),
        )

        # self.render()

        # Extract the rectangle of the screen
        screen_pixels = pygame.surfarray.array3d(self.canvas)
        screen_pixels = np.transpose(np.array(screen_pixels), axes=(1, 0, 2))

        screen_pixels_scaled = pygame.surfarray.array3d(self.screen)
        screen_pixels_scaled = np.transpose(
            np.array(screen_pixels_scaled), axes=(1, 0, 2)
        )
        # print("screen_pixels.shape", screen_pixels.shape)
        # print("screen_pixels_scaled.shape", screen_pixels_scaled.shape)

        observation = screen_pixels[
            # int(top_left.x) : int(bottom_right.x), int(top_left.y) : int(bottom_right.y)
            int(top_left.y) : int(bottom_right.y),
            int(top_left.x) : int(bottom_right.x),
            :,
        ]

        # Resize the observation to a 10x10 image
        if observation.shape != (GRID_SIZE, GRID_SIZE, 3):
            # print("before", observation)
            # observation = np.resize(observation, (GRID_SIZE, GRID_SIZE, 3))
            observation = pygame.transform.smoothscale(
                pygame.surfarray.make_surface(observation), (GRID_SIZE, GRID_SIZE)
            )
            observation = pygame.surfarray.array3d(observation)
            observation = np.transpose(observation, axes=(1, 0, 2))
        # print("after", observation)

        # if player_idx == 0:
        #     # Clear any previous rectangle
        #     # print("after clamp", player_center, top_left, bottom_right)
        #     ax.clear()
        #     # Display the observation
        #     # ax.imshow(observation)
        #     ax.imshow(screen_pixels_scaled)
        #     # Create a rectangle patch
        #     rect = patches.Rectangle(
        #         (top_left.x * SCALE, top_left.y * SCALE),
        #         # (0, 0),
        #         GRID_SIZE * SCALE,
        #         GRID_SIZE * SCALE,  # Rectangle size
        #         linewidth=1,
        #         edgecolor="r",
        #         facecolor="none",
        #     )
        #     # Add the rectangle to the plot
        #     ax.add_patch(rect)
        #     # Draw the plot
        #     fig.canvas.draw_idle()
        #     plt.pause(0.001)

        return observation

    def _clamp_vector(self, vector, min_vector, max_vector):
        return pygame.Vector2(
            max(min_vector.x, min(vector.x, max_vector.x)),
            max(min_vector.y, min(vector.y, max_vector.y)),
        )

    def get_reward(self, player_idx):
        """Returns the reward after an action."""
        return (int(self.check_collisions(player_idx)) - int(self.is_on_black(player_idx))) * 100

    def is_done(self):
        """Checks if the game is finished."""
        return self.steps >= EPISODE_STEPS or self.foods == []

    def reset(self):
        """Resets the game to an initial state."""
        self.foods = [
            Food(
                position=(
                    random.randint(GRID_SIZE // 2, SIZE),
                    random.randint(GRID_SIZE // 2, SIZE),
                ),
                size=FOOD_SIZE,
                color=GREEN,
            )
            for _ in range(FOOD_COUNT)
        ]
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

        observations = [
            self.get_observation(player_idx=i) for i in range(self.num_agents)
        ] * self.num_agents
        self.steps = 0
        return observations

    def render(self, mode="human", scale=1):
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
