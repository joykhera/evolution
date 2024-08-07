import pygame
import random
import math

# Constants
SIZE = 100  # Internal calculation size
SCALE = 5  # Display scale factor
DISPLAY_SIZE = SIZE * SCALE  # Actual size of the window
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
FOOD_COUNT = 50
FPS = 120
FOOD_SIZE = 2
PLAYER_SIZE = 2
PLAYER_SIZE_INCREASE = 0.2
PLAYER_SPEED = 1
DECAY_RATE = 0.001

class Game:
    def __init__(self, size=SIZE, food_count=FOOD_COUNT):
        pygame.init()
        self.size = size
        self.food_count = food_count
        self.screen = pygame.display.set_mode((DISPLAY_SIZE, DISPLAY_SIZE))
        self.clock = pygame.time.Clock()
        self.foods = []
        self.player_pos = [size // 2, size // 2]
        self.player_size = PLAYER_SIZE
        self.running = True

        self.spawn_foods()

    def spawn_foods(self):
        while len(self.foods) < self.food_count:
            new_food = [random.randint(0, self.size - 1), random.randint(0, self.size - 1)]
            self.foods.append(new_food)

    def play(self):
        while self.running:
            for event in pygame.event.get():
                if event.type is pygame.QUIT:
                    self.running = False

            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                self.player_pos[0] = max(0, self.player_pos[0] - PLAYER_SPEED)
            if keys[pygame.K_RIGHT]:
                self.player_pos[0] = min(self.size - 1, self.player_pos[0] + PLAYER_SPEED)
            if keys[pygame.K_UP]:
                self.player_pos[1] = max(0, self.player_pos[1] - PLAYER_SPEED)
            if keys[pygame.K_DOWN]:
                self.player_pos[1] = min(self.size - 1, self.player_pos[1] + PLAYER_SPEED)

            self.update_game_state()
            self.draw()

            self.clock.tick(FPS)
        pygame.quit()

    def update_game_state(self):
        food_eaten = []
        for food in self.foods:
            if self.player_pos[0] <= food[0] < self.player_pos[0] + self.player_size and \
               self.player_pos[1] <= food[1] < self.player_pos[1] + self.player_size:
                food_eaten.append(food)
                self.player_size += PLAYER_SIZE_INCREASE

        for food in food_eaten:
            self.foods.remove(food)
        self.spawn_foods()

        # Apply decay, increasing exponentially with size
        self.player_size *= (1 - DECAY_RATE * math.exp(self.player_size - PLAYER_SIZE))

    def draw(self):
        self.screen.fill(WHITE)
        for food in self.foods:
            pygame.draw.circle(self.screen, GREEN, [int(x * SCALE) for x in food], int(FOOD_SIZE * SCALE))
        pygame.draw.circle(self.screen, RED, [int(x * SCALE) for x in self.player_pos], int(self.player_size * SCALE))
        pygame.display.flip()


if __name__ == "__main__":
    game = Game()
    game.play()
