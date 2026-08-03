import sys
from pathlib import Path
import argparse

sys.path.append(str(Path(__file__).resolve().parents[1]))

from envs.grid_world import CliffWalkingEnv
from exercises.ex1_mdp import ValueIteration
from scripts.ex1_plot import plot_value_function, plot_policy, show_plots


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run value iteration on the Cliff Walking MDP."
    )
    parser.add_argument(
        "--slip_chance",
        type=float,
        default=0.0,
        help="Probability that the executed action differs from the intended action.",
    )
    return parser.parse_args()


def main():
    """Run value iteration on Cliff Walking and save the resulting plots."""
    args = parse_args()
    env = CliffWalkingEnv(slip_chance=args.slip_chance)

    agent = ValueIteration(env, theta=1e-3, gamma=0.9)
    value_fn, policy = agent.value_iteration()

    # Save logs relative to the project root
    project_root = Path(__file__).resolve().parents[1]
    log_dir = project_root / "logs" / "mdp"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Format slip value nicely
    slip_str = f"{args.slip_chance:.2f}".rstrip("0").rstrip(".")

    plot_value_function(
        env,
        value_fn,
        title=f"Value Iteration (slip={slip_str}): State Values",
        save_path=log_dir / f"value_iteration_values_slip_{slip_str}.png",
    )

    plot_policy(
        env,
        policy,
        title=f"Value Iteration (slip={slip_str}): Optimal Policy",
        save_path=log_dir / f"value_iteration_policy_slip_{slip_str}.png",
    )

    show_plots()


if __name__ == "__main__":
    main()