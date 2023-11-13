import pygame
import random
import numpy as np
import neat
import multiprocessing

# Constants
SIZE = 500
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
FOOD_COUNT = 20
FPS = 60
FOOD_SIZE = SIZE // 100
PLAYER_SIZE = SIZE // 50
PLAYER_SIZE_INCREASE = PLAYER_SIZE // 10
PLAYER_SPEED = SIZE // 250
GRID_SIZE = 5


class GameObject:
    """Base class for all game objects."""

    def __init__(self, position, size, color):
        self.position = pygame.Vector2(position)
        self.size = size
        self.color = color

    def draw(self, surface):
        pygame.draw.circle(
            surface,
            self.color,
            (int(self.position.x), int(self.position.y)),
            self.size,  #! change later
        )


class Food(GameObject):
    """Food particles the player can eat."""

    def __init__(self):
        position = (random.randint(0, SIZE), random.randint(0, SIZE))
        super().__init__(position, FOOD_SIZE, GREEN)


class Player(GameObject):
    """Player-controlled agent."""

    def __init__(self, position):
        super().__init__(position, PLAYER_SIZE, RED)
        self.ate = False

    def move(self, direction_vector):
        self.position += direction_vector * PLAYER_SPEED
        self.position.x = max(self.size, min(self.position.x, SIZE - self.size))
        self.position.y = max(self.size, min(self.position.y, SIZE - self.size))


class Game:
    """Main game class that handles game logic."""

    def __init__(self, human_player=True):
        pygame.init()
        self.human_player = (
            human_player  # Determines if the player is human or an ML model
        )
        self.screen = pygame.display.set_mode((SIZE, SIZE))
        pygame.display.set_caption("Evolution Environment")
        self.clock = pygame.time.Clock()
        self.foods = [Food() for _ in range(FOOD_COUNT)]
        # self.player = Player((SIZE // 2, SIZE // 2))
        self.players = []
        self.running = True
        self.steps = 0

    def spawn_player(self):
        self.players.append(Player((SIZE // 2, SIZE // 2)))

    def run(self):
        """Main game loop."""
        while self.running:
            self.screen.fill(WHITE)
            if self.human_player:
                self.handle_events()
            self.update()
            self.render()
            pygame.display.flip()
            self.clock.tick(FPS)
        pygame.quit()

    def handle_events(self):
        """Handles game events, such as input and quitting."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def update(self, action=None):
        """Updates the game state."""
        if self.human_player:
            direction_vector = self.get_input()
        else:
            direction_vector = self.get_action_vector(action)
        self.player.move(direction_vector)
        self.check_collisions()

    def get_action_vector(self, action):
        """Converts an action number into a direction vector."""
        # Define action mappings for ML models
        actions = [
            pygame.Vector2(0, 0),  # None
            pygame.Vector2(1, 0),  # Right
            pygame.Vector2(-1, 0),  # Left
            pygame.Vector2(0, -1),  # Up
            pygame.Vector2(0, 1),  # Down
        ]
        return actions[action] if action is not None else pygame.Vector2(0, 0)

    def play_step(self, action):
        """Performs a single step in the game loop."""
        self.screen.fill(WHITE)
        self.update(action)
        self.render()
        pygame.display.flip()
        reward = self.get_reward()
        done = self.is_done()
        self.clock.tick(FPS)
        self.steps += 1
        return self.get_observation(), reward, done

    def render(self):
        """Renders all game objects to the screen."""
        for food in self.foods:
            food.draw(self.screen)
        self.player.draw(self.screen)

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

    def check_collisions(self):
        """Checks for collisions between player and food."""
        for food in self.foods[:]:
            if self.player.position.distance_to(food.position) < self.player.size:
                print("Collision!", self.player.size)
                self.foods.remove(food)
                self.ate = True
                # self.player.size += PLAYER_SIZE_INCREASE
                # self.foods.append(Food())

    def get_observation(self):
        """Returns the current observation of the game state."""
        # Get a 100x100 pixel array around the player
        # This requires additional implementation to extract the pixels around the player
        player_center = self.player.position
        top_left = player_center - pygame.Vector2(GRID_SIZE // 2, GRID_SIZE // 2)
        bottom_right = player_center + pygame.Vector2(GRID_SIZE // 2, GRID_SIZE // 2)
        # Make sure we don't go out of bounds
        top_left = self._clamp_vector(
            top_left,
            pygame.Vector2(0, 0),
            pygame.Vector2(SIZE, SIZE) - pygame.Vector2(1, 1),
        )
        bottom_right = self._clamp_vector(
            bottom_right,
            pygame.Vector2(0, 0),
            pygame.Vector2(SIZE, SIZE) - pygame.Vector2(1, 1),
        )
        # Extract the rectangle of the screen
        screen_pixels = pygame.surfarray.array3d(self.screen)
        observation = screen_pixels[
            int(top_left.x) : int(bottom_right.x), int(top_left.y) : int(bottom_right.y)
        ]
        # Resize the observation to a 100x100 image if necessary
        observation = np.resize(observation, (GRID_SIZE, GRID_SIZE, 3))
        return observation

    def _clamp_vector(self, vector, min_vector, max_vector):
        return pygame.Vector2(
            max(min_vector.x, min(vector.x, max_vector.x)),
            max(min_vector.y, min(vector.y, max_vector.y)),
        )

    def get_reward(self):
        """Returns the reward after an action."""
        # Example reward function: reward the player for eating food
        # You can modify this as needed for your game
        return self.player.ate

    def is_done(self):
        """Checks if the game is finished."""
        # Example condition: game ends when player reaches a certain size
        # You can modify this as needed for your game
        return self.steps >= 1000

    def reset(self):
        """Resets the game to an initial state."""
        self.foods = [Food() for _ in range(FOOD_COUNT)]
        self.player = Player((SIZE // 2, SIZE // 2))
        # self.player.size = PLAYER_SIZE  # Reset player size or any other variables
        # Reset other game state variables if necessary

    def render(self, mode="human"):
        """Renders the game to the Pygame window."""
        if mode == "human":
            self.screen.fill(WHITE)
            for food in self.foods:
                food.draw(self.screen)
            self.player.draw(self.screen)
            pygame.display.flip()

    def close(self):
        """Closes the Pygame window."""
        pygame.quit()

    def output_to_action(self, output):
        # Convert the neural network output to a game action
        return output.index(max(output))
    


def run_game_with_human():
    game = Game(human_player=True)
    game.run()


# This allows the file to be imported without running the game immediately
if __name__ == "__main__":
    run_game_with_human()
