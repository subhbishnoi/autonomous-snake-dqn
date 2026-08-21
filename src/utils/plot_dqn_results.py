import pandas as pd
import matplotlib.pyplot as plt


RESULTS_PATH = "results/dqn_results.csv"
OUTPUT_PATH = "results/dqn_training.png"


def plot_dqn_results():

    df = pd.read_csv(RESULTS_PATH)

    df["moving_average"] = (
        df["score"]
        .rolling(window=10)
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
        label="10-Episode Moving Average",
    )

    plt.xlabel("Episode")
    plt.ylabel("Score")

    plt.title(
        "DQN Snake Training"
    )

    plt.legend()
    plt.grid(alpha=0.2)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_PATH,
        dpi=200,
    )

    plt.show()

    print(
        f"Plot saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    plot_dqn_results()