import numpy as np
import pygame


class Agent:
    def __init__(self, position, size, speed, color, map_size, grid_size):
        self.position = np.array(position, dtype=np.float32)
        self.size = size
        self.speed = speed
        self.color = color
        self.map_size = map_size
        self.grid_size = grid_size

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

    def draw(self, screen, scale, draw_grid=False):
        position = (self.position * scale).astype(int)
        pygame.draw.circle(screen, self.color, position, self.size * scale)

        if draw_grid:
            half_grid = 5 * scale
            pygame.draw.rect(screen, self.color, (position[0] - half_grid, position[1] - half_grid, 2 * half_grid, 2 * half_grid), 1)

    def get_observation(self, observation_canvas):
        x = self.position[0]
        y = self.position[1]
        half_grid = self.grid_size // 2

        # Ensure the observation window stays within the observation canvas boundaries
        top_left_x = max(x - half_grid, 0)
        top_left_y = max(y - half_grid, 0)
        bottom_right_x = min(x + half_grid, self.map_size)
        bottom_right_y = min(y + half_grid, self.map_size)

        width = bottom_right_x - top_left_x
        height = bottom_right_y - top_left_y

        if width > 0 and height > 0:
            # Create a surface for the agent's field of view
            sub_surface = pygame.Surface((width, height))
            sub_surface.blit(observation_canvas, (0, 0), pygame.Rect(top_left_x, top_left_y, width, height))
            sub_surface = pygame.transform.scale(sub_surface, (self.grid_size, self.grid_size))
            observation = pygame.surfarray.array3d(sub_surface)
            observation = np.transpose(observation, (1, 0, 2))  # Convert to (grid_size, grid_size, 3)

            # Flatten the observation to a 2D array of tuples for fast comparison
            flat_observation = observation.reshape(-1, 3)

            # Define the color mapping
            color_map = {
                (0, 0, 0): 0,        # Black
                (255, 255, 255): 1,  # White
                (0, 255, 0): 2,      # Green
                (255, 0, 0): 3,      # Red
                (0, 0, 255): 4,      # Blue
            }

            # Create an array of zeros
            discrete_observation = np.zeros(flat_observation.shape[0], dtype=np.uint8)

            # Vectorized comparison and assignment
            for color, value in color_map.items():
                mask = np.all(flat_observation == color, axis=-1)
                discrete_observation[mask] = value

            # Reshape the flat array back to the grid size
            discrete_observation = discrete_observation.reshape(self.grid_size, self.grid_size)

            return discrete_observation
        else:
            # Return a blank observation if something goes wrong
            return np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)

