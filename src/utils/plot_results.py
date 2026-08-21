import pandas as pd
import matplotlib.pyplot as plt


RESULTS_PATH = "results/q_learning_results.csv"
OUTPUT_PATH = "results/q_learning_training.png"


def plot_training_results():

    df = pd.read_csv(RESULTS_PATH)

    # Calculate moving average over 50 episodes.
    df["moving_average"] = (
        df["score"]
        .rolling(window=50)
        .mean()
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        df["episode"],
        df["score"],
        alpha=0.3,
        label="Episode Score",
    )

    plt.plot(
        df["episode"],
        df["moving_average"],
        linewidth=2,
        label="50-Episode Moving Average",
    )

    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title("Q-Learning Snake Training")

    plt.legend()
    plt.grid(alpha=0.2)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_PATH,
        dpi=200,
    )

    plt.show()

    print(f"Plot saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    plot_training_results()