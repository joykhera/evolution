import pygame


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
