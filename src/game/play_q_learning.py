import os
import pickle
import time

import pygame

from src.agents.q_learning_agent import QLearningAgent
from src.game.environment import SnakeEnvironment
from src.game.config import (
    GRID_WIDTH,
    GRID_HEIGHT,
    CELL_SIZE,
    FPS,
)


Q_TABLE_PATH = "models/q_learning/q_table.pkl"


def draw_game(screen, env, score, steps):
    """Draw the current Snake game state."""

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
            color = (80, 220, 100)
        else:
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
    # Update window
    # --------------------------------------------------

    pygame.display.set_caption(
        f"Q-Learning Snake | Score: {score} | Steps: {steps}"
    )

    pygame.display.flip()


def load_q_table(agent, path):
    """Load a trained Q-table into the agent."""

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Q-table not found: {path}\n"
            "Make sure Q-Learning training has been completed."
        )

    with open(path, "rb") as file:
        saved_q_table = pickle.load(file)

    agent.q_table = saved_q_table

    print(f"Loaded Q-table: {path}")
    print(f"Q-table states: {len(agent.q_table)}")


def main():

    print("=" * 50)
    print("AUTONOMOUS SNAKE - Q-LEARNING")
    print("=" * 50)

    # --------------------------------------------------
    # Create environment
    # --------------------------------------------------

    env = SnakeEnvironment()

    # --------------------------------------------------
    # Create Q-Learning agent
    # --------------------------------------------------

    agent = QLearningAgent(
        state_size=11,
        action_size=3,
    )

    # --------------------------------------------------
    # Load trained Q-table
    # --------------------------------------------------

    load_q_table(
        agent,
        Q_TABLE_PATH,
    )

    # --------------------------------------------------
    # Disable exploration
    # --------------------------------------------------

    agent.epsilon = 0.0

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
        "Q-Learning Snake"
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

        # Handle Pygame events
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

        # --------------------------------------------------
        # Let Q-Learning agent choose action
        # --------------------------------------------------

        if not game_over:

            action = agent.choose_action(state)

            next_state, reward, done = env.step(action)

            state = next_state

            steps += 1

            score = env.get_score()

            game_over = done

        # --------------------------------------------------
        # Render game
        # --------------------------------------------------

        draw_game(
            screen,
            env,
            score,
            steps,
        )

        clock.tick(FPS)

        # --------------------------------------------------
        # Game over
        # --------------------------------------------------

        if game_over:

            pygame.display.set_caption(
                f"Q-Learning Snake | GAME OVER | Score: {score}"
            )

            print()
            print("=" * 50)
            print("GAME OVER")
            print(f"Final score: {score}")
            print(f"Total steps: {steps}")
            print("=" * 50)

            time.sleep(2)

            running = False

    pygame.quit()


if __name__ == "__main__":
    main()