"""
Evaluation script for DQN on CartPole-v1.
Supports both quantitative evaluation and GUI playback.
"""

import sys
from pathlib import Path
import random
import argparse
import numpy as np
import torch
import gymnasium as gym

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from envs.cartpole_wrapper import CartPoleWrapper
from exercises.ex2_dqn import DQN
from exercises.ex2_dqn_config import DQN_PARAMETERS


def evaluate_policy(env, agent, num_episodes):
    """Evaluate a trained policy for several episodes."""
    returns = []
    lengths = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_return = 0.0
        episode_length = 0

        while not done:
            action = agent.predict_action(state)
            next_state, reward, done, _ = env.step(action)

            state = next_state
            episode_return += reward
            episode_length += 1

        returns.append(float(episode_return))
        lengths.append(int(episode_length))

        print(
            f"Eval Episode {episode + 1:02d} | "
            f"Return: {episode_return:.1f} | Length: {episode_length}"
        )

    return returns, lengths


def summarize_metrics(returns, lengths, success_threshold):
    """Compute summary metrics for evaluation."""
    returns_np = np.array(returns, dtype=np.float32)
    lengths_np = np.array(lengths, dtype=np.int32)

    metrics = {
        "num_episodes": int(len(returns)),
        "mean_return": float(np.mean(returns_np)),
        "std_return": float(np.std(returns_np)),
        "min_return": float(np.min(returns_np)),
        "max_return": float(np.max(returns_np)),
        "median_return": float(np.median(returns_np)),
        "mean_length": float(np.mean(lengths_np)),
        "std_length": float(np.std(lengths_np)),
        "min_length": int(np.min(lengths_np)),
        "max_length": int(np.max(lengths_np)),
        "success_threshold": float(success_threshold),
        "success_rate": float(np.mean(returns_np >= success_threshold)),
        "returns": returns,
        "lengths": lengths,
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate or play a trained DQN policy on CartPole-v1."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(ROOT_DIR / "logs" / "dqn" / "models" / "dqn_cartpole.pth"),
        help="Path to the trained DQN checkpoint.",
    )
    parser.add_argument(
        "--num_eval_episodes",
        type=int,
        default=50,
        help="Number of evaluation episodes.",
    )
    parser.add_argument(
        "--success_threshold",
        type=float,
        default=475.0,
        help="Return threshold used to compute success rate.",
    )
    parser.add_argument(
        "--record_video",
        action="store_true",
        help="Record a video for the first evaluation episode.",
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Open a GUI window and play the learned policy.",
    )
    args = parser.parse_args()

    if args.play and args.record_video:
        raise ValueError("--play and --record_video cannot be used at the same time.")

    # Hyperparameters
    hidden_dim = DQN_PARAMETERS["hidden_dim"]
    seed = DQN_PARAMETERS["seed"]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU name: {torch.cuda.get_device_name(0)}")

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    # Render mode
    if args.play:
        render_mode = "human"
    elif args.record_video:
        render_mode = "rgb_array"
    else:
        render_mode = None

    env = CartPoleWrapper(seed=seed, render_mode=render_mode)

    # Video recording
    if args.record_video:
        video_dir = ROOT_DIR / "logs" / "dqn" / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)

        env.env = gym.wrappers.RecordVideo(
            env.env,
            video_folder=str(video_dir),
            episode_trigger=lambda episode_id: episode_id == 0,
            name_prefix="dqn_cartpole_eval",
        )
        print(f"Video will be saved to: {video_dir}")

    if args.play:
        print("Play mode enabled: opening GUI window...")

    # Agent
    agent = DQN(
        state_dim=env.state_dim,
        hidden_dim=hidden_dim,
        action_dim=env.action_dim,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon=0.0,
        target_update=100,
        device=device,
    )

    agent.load(str(model_path))
    print(f"Loaded checkpoint from: {model_path}")

    # Evaluation
    returns, lengths = evaluate_policy(
        env=env,
        agent=agent,
        num_episodes=args.num_eval_episodes,
    )

    env.close()

    metrics = summarize_metrics(
        returns=returns,
        lengths=lengths,
        success_threshold=args.success_threshold,
    )

    print("\n===== Evaluation Summary =====")
    print(f"Number of episodes : {metrics['num_episodes']}")
    print(f"Mean return        : {metrics['mean_return']:.2f}")
    print(f"Std return         : {metrics['std_return']:.2f}")
    print(f"Min return         : {metrics['min_return']:.2f}")
    print(f"Max return         : {metrics['max_return']:.2f}")
    print(f"Median return      : {metrics['median_return']:.2f}")
    print(f"Mean length        : {metrics['mean_length']:.2f}")
    print(f"Std length         : {metrics['std_length']:.2f}")
    print(f"Success threshold  : {metrics['success_threshold']:.1f}")
    print(f"Success rate       : {metrics['success_rate'] * 100:.1f}%")


if __name__ == "__main__":
    main()