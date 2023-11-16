import pygame


class GameObject:
    """Base class for all game objects."""

    def __init__(self, position, size, color):
        self.position = pygame.Vector2(position)
        self.size = size
        self.color = color

    def draw(self, surface, scale=1):
        pygame.draw.circle(
            surface,
            self.color,
            (int(self.position.x * scale), int(self.position.y * scale)),
            self.size * scale,  #! change later
        )
