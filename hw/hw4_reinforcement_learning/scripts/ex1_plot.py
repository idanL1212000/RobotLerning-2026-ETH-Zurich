import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


ACTION_ARROWS = {
    0: "↑",
    1: "↓",
    2: "←",
    3: "→",
}


def plot_value_function(env, value_fn, title="State Value Function", save_path=None):
    grid = value_fn.reshape(env.nrow, env.ncol)

    plt.figure(figsize=(12, 3))
    plt.imshow(grid, cmap="viridis", origin="upper")
    plt.colorbar(label="Value")

    for r in range(env.nrow):
        for c in range(env.ncol):
            s = r * env.ncol + c
            if s in env.cliff_states:
                continue

            plt.text(
                c, r,
                f"{grid[r, c]:.1f}",
                ha="center",
                va="center",
                color="white",
                fontsize=9,
            )

    plt.title(title)
    plt.xticks(range(env.ncol))
    plt.yticks(range(env.nrow))
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)


def plot_policy(env, policy, title="Optimal Policy", save_path=None):
    plt.figure(figsize=(12, 3))

    grid = np.zeros((env.nrow, env.ncol), dtype=int)
    for s in env.cliff_states:
        r, c = env.state_to_pos(s)
        grid[r, c] = 1

    cmap = ListedColormap(["#EAEAEA", "#000000"])
    plt.imshow(grid, cmap=cmap, origin="upper")

    for r in range(env.nrow):
        for c in range(env.ncol):
            s = r * env.ncol + c

            if s in env.cliff_states:
                continue

            if s == env.start_state:
                plt.text(c, r, "S", ha="center", va="center",
                         fontsize=14, fontweight="bold", color="green")
                continue

            if s == env.goal_state:
                plt.text(c, r, "G", ha="center", va="center",
                         fontsize=14, fontweight="bold", color="red")
                continue

            action_probs = policy[s]
            best_actions = np.flatnonzero(np.isclose(action_probs, action_probs.max()))
            arrow_text = "".join(ACTION_ARROWS[a] for a in best_actions)

            plt.text(c, r, arrow_text, ha="center", va="center",
                     fontsize=14, color="blue")

    plt.title(title)
    plt.xticks(range(env.ncol))
    plt.yticks(range(env.nrow))

    ax = plt.gca()
    ax.set_xticks(np.arange(-0.5, env.ncol, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.nrow, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)


def show_plots():
    plt.show()