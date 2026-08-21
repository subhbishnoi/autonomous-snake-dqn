import time

import pygame

from src.agents.dqn_agent import DQNAgent
from src.game.environment import SnakeEnvironment
from src.game.config import (
    GRID_WIDTH,
    GRID_HEIGHT,
    CELL_SIZE,
    FPS,
)


MODEL_PATH = "models/dqn/snake_dqn.keras"


def draw_game(screen, env, score, steps):
    """Draw the current Snake game state."""

    # Clear screen
    screen.fill((20, 20, 20))

    # --------------------------------------------------
    # Draw food
    # --------------------------------------------------

    food_x, food_y = env.snake.food

    pygame.draw.rect(
        screen,
        (220, 60, 60),
        (
            food_x * CELL_SIZE,
            food_y * CELL_SIZE,
            CELL_SIZE,
            CELL_SIZE,
        ),
    )

    # --------------------------------------------------
    # Draw snake
    # --------------------------------------------------

    for index, (x, y) in enumerate(env.snake.body):

        if index == 0:
            # Snake head
            color = (80, 220, 100)
        else:
            # Snake body
            color = (40, 160, 70)

        pygame.draw.rect(
            screen,
            color,
            (
                x * CELL_SIZE,
                y * CELL_SIZE,
                CELL_SIZE,
                CELL_SIZE,
            ),
        )

    # --------------------------------------------------
    # Draw grid
    # --------------------------------------------------

    for x in range(GRID_WIDTH):

        pygame.draw.line(
            screen,
            (45, 45, 45),
            (x * CELL_SIZE, 0),
            (
                x * CELL_SIZE,
                GRID_HEIGHT * CELL_SIZE,
            ),
        )

    for y in range(GRID_HEIGHT):

        pygame.draw.line(
            screen,
            (45, 45, 45),
            (0, y * CELL_SIZE),
            (
                GRID_WIDTH * CELL_SIZE,
                y * CELL_SIZE,
            ),
        )

    # --------------------------------------------------
    # Update window title
    # --------------------------------------------------

    pygame.display.set_caption(
        f"DQN Snake | Score: {score} | Steps: {steps}"
    )

    # IMPORTANT:
    # Actually update the Pygame window.
    pygame.display.flip()


def main():

    print("=" * 50)
    print("AUTONOMOUS SNAKE - DQN")
    print("=" * 50)

    # --------------------------------------------------
    # Create environment
    # --------------------------------------------------

    env = SnakeEnvironment()

    # --------------------------------------------------
    # Create DQN agent
    # --------------------------------------------------

    agent = DQNAgent(
        state_size=11,
        action_size=3,
        learning_rate=0.001,
        gamma=0.95,
        epsilon=0.0,
        epsilon_min=0.0,
        epsilon_decay=1.0,
        replay_capacity=100_000,
        batch_size=64,
        target_update_frequency=500,
    )

    # --------------------------------------------------
    # Load trained model
    # --------------------------------------------------

    agent.load(MODEL_PATH)

    # Disable exploration
    agent.epsilon = 0.0

    print(f"Loaded model: {MODEL_PATH}")
    print("Exploration: OFF")
    print("Starting autonomous Snake...")

    # --------------------------------------------------
    # Initialize Pygame
    # --------------------------------------------------

    pygame.init()

    screen = pygame.display.set_mode(
        (
            GRID_WIDTH * CELL_SIZE,
            GRID_HEIGHT * CELL_SIZE,
        )
    )

    pygame.display.set_caption(
        "DQN Snake"
    )

    clock = pygame.time.Clock()

    # --------------------------------------------------
    # Reset environment
    # --------------------------------------------------

    state = env.reset()

    score = 0
    steps = 0

    running = True
    game_over = False

    # --------------------------------------------------
    # Main game loop
    # --------------------------------------------------

    while running:

        # ----------------------------------------------
        # Handle Pygame events
        # ----------------------------------------------

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

        # ----------------------------------------------
        # Let DQN play
        # ----------------------------------------------

        if not game_over:

            # DQN selects action
            action = agent.choose_action(state)

            # Environment executes action
            next_state, reward, done = env.step(action)

            # Move to next state
            state = next_state

            steps += 1

            score = env.get_score()

            game_over = done

        # ----------------------------------------------
        # Draw game
        # ----------------------------------------------

        draw_game(
            screen,
            env,
            score,
            steps,
        )

        # ----------------------------------------------
        # Control game speed
        # ----------------------------------------------

        clock.tick(FPS)

        # ----------------------------------------------
        # Game over
        # ----------------------------------------------

        if game_over:

            pygame.display.set_caption(
                f"DQN Snake | GAME OVER | Score: {score}"
            )

            print()
            print("=" * 50)
            print("GAME OVER")
            print(f"Final score: {score}")
            print(f"Total steps: {steps}")
            print("=" * 50)

            # Keep final screen visible
            time.sleep(2)

            running = False

    # --------------------------------------------------
    # Close Pygame
    # --------------------------------------------------

    pygame.quit()


if __name__ == "__main__":
    main()