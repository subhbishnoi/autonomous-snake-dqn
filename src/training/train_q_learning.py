import os
import pickle

import numpy as np
import pandas as pd

from src.agents.q_learning_agent import QLearningAgent
from src.game.environment import SnakeEnvironment
from src.game.config import NUM_EPISODES


def train_q_learning(num_episodes=NUM_EPISODES):
    """
    Train the Q-Learning agent on the Snake environment.
    """

    env = SnakeEnvironment()
    agent = QLearningAgent()

    results = []

    for episode in range(1, num_episodes + 1):

        state = env.reset()

        total_reward = 0
        done = False
        steps = 0

        while not done:

            # Agent chooses an action.
            action = agent.choose_action(state)

            # Environment executes action.
            next_state, reward, done = env.step(action)

            # Agent learns from the experience.
            agent.update(
                state,
                action,
                reward,
                next_state,
                done,
            )

            state = next_state

            total_reward += reward
            steps += 1

        # Reduce exploration after each episode.
        agent.decay_epsilon()

        score = env.get_score()

        results.append(
            {
                "episode": episode,
                "score": score,
                "total_reward": total_reward,
                "steps": steps,
                "epsilon": agent.epsilon,
                "q_table_size": agent.get_q_table_size(),
            }
        )

        # Print progress.
        if episode % 10 == 0 or episode == 1:

            recent_scores = [
                result["score"]
                for result in results[-100:]
            ]

            average_score = np.mean(recent_scores)

            print(
                f"Episode: {episode:4d} | "
                f"Score: {score:3d} | "
                f"Avg Score: {average_score:.2f} | "
                f"Reward: {total_reward:6.1f} | "
                f"Epsilon: {agent.epsilon:.4f} | "
                f"States: {agent.get_q_table_size()}"
            )

    return agent, results


def save_results(results):
    """Save training results to CSV."""

    os.makedirs("results", exist_ok=True)

    df = pd.DataFrame(results)

    output_path = "results/q_learning_results.csv"

    df.to_csv(output_path, index=False)

    print(f"\nTraining results saved to: {output_path}")


def save_q_table(agent):
    """Save the trained Q-table."""

    os.makedirs("models/q_learning", exist_ok=True)

    output_path = "models/q_learning/q_table.pkl"

    with open(output_path, "wb") as file:
        pickle.dump(agent.q_table, file)

    print(f"Q-table saved to: {output_path}")
    print(f"Q-table states: {len(agent.q_table)}")


if __name__ == "__main__":

    agent, results = train_q_learning()

    save_results(results)

    save_q_table(agent)