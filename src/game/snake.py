import random

from src.game.config import (
    GRID_WIDTH,
    GRID_HEIGHT,
)


class Snake:
    def __init__(self):
        self.reset()

    def reset(self):
        """Reset the snake to its initial state."""
        center_x = GRID_WIDTH // 2
        center_y = GRID_HEIGHT // 2

        self.body = [
            (center_x, center_y),
            (center_x - 1, center_y),
            (center_x - 2, center_y),
        ]

        self.direction = (1, 0)
        self.score = 0

        self.food = self._generate_food()

    def _generate_food(self):
        """Generate food at a position not occupied by the snake."""
        available_positions = [
            (x, y)
            for x in range(GRID_WIDTH)
            for y in range(GRID_HEIGHT)
            if (x, y) not in self.body
        ]

        return random.choice(available_positions)

    def move(self):
        """Move the snake one cell in its current direction."""

        head_x, head_y = self.body[0]
        direction_x, direction_y = self.direction

        new_head = (
            head_x + direction_x,
            head_y + direction_y,
        )

        self.body.insert(0, new_head)

        ate_food = new_head == self.food

        if ate_food:
            self.score += 1
            self.food = self._generate_food()
        else:
            self.body.pop()

        return ate_food

    def set_direction(self, direction):
        """Change the snake's movement direction."""
        self.direction = direction

    def get_head(self):
        """Return the snake's current head position."""
        return self.body[0]

    def check_collision(self):
        """Check whether the snake has collided with a wall or itself."""
        head_x, head_y = self.get_head()

        # Wall collision
        if (
            head_x < 0
            or head_x >= GRID_WIDTH
            or head_y < 0
            or head_y >= GRID_HEIGHT
        ):
            return True

        # Self collision
        if self.get_head() in self.body[1:]:
            return True

        return False