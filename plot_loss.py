"""
Plot loss convergence curve from the inverse rendering optimization.
Run after diff_pathtracer.py has completed.
Requires matplotlib: pip install matplotlib
"""

import sys

import numpy as np


def plot_matplotlib(loss_history):
    """Matplotlib loss curve saved to output/loss_curve.png."""
    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(loss_history, color="#e74c3c", linewidth=2)
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("L2 Loss", fontsize=12)
    ax.set_title("Inverse Rendering — Loss Convergence", fontsize=14)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("output/loss_curve.png", dpi=150)
    plt.show()
    print("  Saved output/loss_curve.png")


if __name__ == "__main__":
    loss_history = np.load("output/loss_history.npy")

    try:
        plot_matplotlib(loss_history)
    except ImportError:
        print("  (matplotlib not available — skipping PNG plot)")
