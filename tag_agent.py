import numpy as np
import pygame


class Agent:
    def __init__(self, position, size, speed, color, color_encoding, map_size, grid_size, scale):
        self.position = np.array(position, dtype=np.float32)
        self.size = size
        self.speed = speed
        self.color = color
        self.color_encoding = color_encoding
        self.map_size = map_size
        self.grid_size = grid_size
        self.scale = scale

    def move(self, action):
        # print(action, self.position)
        if action == 1:  # move left
            self.position[0] = max(self.position[0] - self.speed, 0)
        elif action == 2:  # move right
            self.position[0] = min(self.position[0] + self.speed, self.map_size)
        elif action == 3:  # move down
            self.position[1] = max(self.position[1] - self.speed, 0)
        elif action == 4:  # move up
            self.position[1] = min(self.position[1] + self.speed, self.map_size)

    def draw(self, screen, render_mode=None, draw_grid=False):
        if render_mode == "human":
            position = (self.position * self.scale).astype(int)
            pygame.draw.circle(screen, self.color, position, self.size * self.scale)

            if draw_grid:
                half_grid = 5 * self.scale
                pygame.draw.rect(screen, self.color, (position[0] - half_grid, position[1] - half_grid, 2 * half_grid, 2 * half_grid), 1)
        else:
            position = self.position
            pygame.draw.circle(screen, self.color_encoding, position, self.size)

    def get_observation(self, observation_canvas):
        x, y = self.position.astype(int)
        half_grid = self.grid_size // 2

        # Determine the observation window bounds within the map
        top_left_x = max(x - half_grid, 0)
        top_left_y = max(y - half_grid, 0)
        bottom_right_x = min(x + half_grid, self.map_size)
        bottom_right_y = min(y + half_grid, self.map_size)

        # Calculate the width and height of the valid observation area
        width = bottom_right_x - top_left_x
        height = bottom_right_y - top_left_y

        # Initialize a blank observation grid with zeros
        observation_array = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)

        if width > 0 and height > 0:
            # Directly extract the relevant section of the observation canvas
            observation = pygame.surfarray.pixels2d(observation_canvas)[top_left_x:bottom_right_x, top_left_y:bottom_right_y]

            # Determine where to place the observation in the grid
            start_x = half_grid - (x - top_left_x)
            start_y = half_grid - (y - top_left_y)

            # Resize the observation to fit into the grid
            observation_surface = pygame.transform.scale(pygame.surfarray.make_surface(observation), (width, height))

            # Convert the surface back to a 2D numpy array
            observation_resized = pygame.surfarray.pixels2d(observation_surface)

            # Place the resized observation in the appropriate location in the observation grid
            observation_array[start_x:start_x + width, start_y:start_y + height] = observation_resized
            observation_array_normalized = observation_array / 4.0
        # print(observation_array, observation_array_normalized)

        return observation_array_normalized
