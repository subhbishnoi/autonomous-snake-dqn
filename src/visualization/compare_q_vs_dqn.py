import os

import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# Paths
# --------------------------------------------------

Q_RESULTS = "results/q_learning_results.csv"
DQN_RESULTS = "results/dqn_results.csv"

OUTPUT_DIR = "results/plots"


# --------------------------------------------------
# Load results
# --------------------------------------------------

q_df = pd.read_csv(Q_RESULTS)
dqn_df = pd.read_csv(DQN_RESULTS)


# --------------------------------------------------
# Create output directory
# --------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# 1. Score comparison
# --------------------------------------------------

plt.figure(figsize=(12, 6))

plt.plot(
    q_df["episode"],
    q_df["score"],
    alpha=0.25,
    label="Q-Learning"
)

plt.plot(
    dqn_df["episode"],
    dqn_df["score"],
    alpha=0.25,
    label="DQN"
)

# Rolling average
q_rolling = q_df["score"].rolling(
    window=50
).mean()

dqn_rolling = dqn_df["score"].rolling(
    window=50
).mean()

plt.plot(
    q_df["episode"],
    q_rolling,
    linewidth=2,
    label="Q-Learning (50-episode avg)"
)

plt.plot(
    dqn_df["episode"],
    dqn_rolling,
    linewidth=2,
    label="DQN (50-episode avg)"
)

plt.xlabel("Episode")
plt.ylabel("Score")

plt.title(
    "Q-Learning vs DQN - Score During Training"
)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/q_vs_dqn_score.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 2. Average score comparison
# --------------------------------------------------

q_mean = q_df["score"].mean()
dqn_mean = dqn_df["score"].mean()

plt.figure(figsize=(8, 6))

plt.bar(
    ["Q-Learning", "DQN"],
    [q_mean, dqn_mean]
)

plt.ylabel("Mean Score")

plt.title(
    "Average Score Comparison"
)

for i, value in enumerate(
    [q_mean, dqn_mean]
):
    plt.text(
        i,
        value,
        f"{value:.2f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/average_score_comparison.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 3. Best 100-episode performance
# --------------------------------------------------

q_best_100 = (
    q_df["score"]
    .rolling(100)
    .mean()
    .max()
)

dqn_best_100 = (
    dqn_df["score"]
    .rolling(100)
    .mean()
    .max()
)

plt.figure(figsize=(8, 6))

plt.bar(
    ["Q-Learning", "DQN"],
    [q_best_100, dqn_best_100]
)

plt.ylabel("Best 100-Episode Average")

plt.title(
    "Best 100-Episode Performance"
)

for i, value in enumerate(
    [q_best_100, dqn_best_100]
):
    plt.text(
        i,
        value,
        f"{value:.2f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/best_100_comparison.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# Finished
# --------------------------------------------------

print(
    "\nComparison graphs created successfully!"
)

print(
    f"Saved to: {OUTPUT_DIR}"
)

print(
    "\nGenerated files:"
)

print(
    "1. q_vs_dqn_score.png"
)

print(
    "2. average_score_comparison.png"
)

print(
    "3. best_100_comparison.png"
)