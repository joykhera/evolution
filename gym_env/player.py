from gym_env.game_object import GameObject


class Player(GameObject):
    """Player-controlled agent."""

    def __init__(self, position, map_size, size, speed, color, boost_color, max_speed, max_boost):
        super().__init__(position, size, color)
        self.ate = False
        self.map_size = map_size
        self.base_speed = speed
        self.current_speed = speed
        self.max_speed = max_speed
        self.max_boost = max_boost
        self.cur_boost = max_boost
        self.boost_active = False
        self.boost_color = boost_color
        self.normal_color = color

    def move(self, direction_vector, boost=False):
        if boost and self.cur_boost > 0:
            self.boost_active = True
            self.color = self.boost_color
            self.current_speed = self.max_speed
            self.cur_boost -= 1
        else:
            self.boost_active = False
            self.color = self.normal_color
            self.current_speed = self.base_speed
            if self.cur_boost < self.max_boost:
                self.cur_boost += 1

        self.position += direction_vector * self.current_speed
        self.position.x = max(0, min(self.position.x, self.map_size))
        self.position.y = max(0, min(self.position.y, self.map_size))
