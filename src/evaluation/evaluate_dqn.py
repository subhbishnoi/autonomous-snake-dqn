import os

import numpy as np
import pandas as pd

from src.agents.dqn_agent import DQNAgent
from src.game.environment import SnakeEnvironment


MODEL_PATH = "models/dqn/snake_dqn.keras"
OUTPUT_PATH = "results/dqn_evaluation.csv"

NUM_EVALUATION_EPISODES = 100


def evaluate_dqn():

    print("=" * 60)
    print("DQN EVALUATION")
    print("=" * 60)

    # ------------------------------------------
    # Create environment
    # ------------------------------------------

    env = SnakeEnvironment()

    # ------------------------------------------
    # Create agent
    # ------------------------------------------

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

    # ------------------------------------------
    # Load trained model
    # ------------------------------------------

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    agent.load(MODEL_PATH)

    # Make absolutely sure the agent
    # does not explore during evaluation.
    agent.epsilon = 0.0

    print(f"Loaded model: {MODEL_PATH}")
    print(f"Evaluation episodes: {NUM_EVALUATION_EPISODES}")
    print("Exploration: OFF")
    print("=" * 60)

    results = []

    # ------------------------------------------
    # Evaluation loop
    # ------------------------------------------

    for episode in range(
        1,
        NUM_EVALUATION_EPISODES + 1,
    ):

        state = env.reset()

        total_reward = 0
        steps = 0

        done = False

        while not done:

            # Greedy action.
            action = agent.choose_action(state)

            next_state, reward, done = env.step(
                action
            )

            state = next_state

            total_reward += reward
            steps += 1

        score = env.get_score()

        results.append(
            {
                "episode": episode,
                "score": score,
                "total_reward": total_reward,
                "steps": steps,
            }
        )

        if (
            episode == 1
            or episode % 10 == 0
        ):

            print(
                f"Episode: {episode:3d} | "
                f"Score: {score:3d} | "
                f"Reward: {total_reward:8.2f} | "
                f"Steps: {steps:4d}"
            )

    # ------------------------------------------
    # Save results
    # ------------------------------------------

    df = pd.DataFrame(results)

    os.makedirs(
        "results",
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # ------------------------------------------
    # Calculate metrics
    # ------------------------------------------

    average_score = df["score"].mean()

    max_score = df["score"].max()

    median_score = df["score"].median()

    average_reward = (
        df["total_reward"].mean()
    )

    average_steps = df["steps"].mean()

    food_rate = (
        (df["score"] > 0).mean() * 100
    )

    print("\n")
    print("=" * 60)
    print("FINAL DQN EVALUATION")
    print("=" * 60)

    print(
        f"Average score:       {average_score:.2f}"
    )

    print(
        f"Maximum score:       {max_score:.0f}"
    )

    print(
        f"Median score:        {median_score:.2f}"
    )

    print(
        f"Average reward:      {average_reward:.2f}"
    )

    print(
        f"Average survival:    {average_steps:.2f}"
    )

    print(
        f"Food acquisition:    {food_rate:.2f}%"
    )

    print("=" * 60)

    print(
        f"\nEvaluation results saved to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    evaluate_dqn()