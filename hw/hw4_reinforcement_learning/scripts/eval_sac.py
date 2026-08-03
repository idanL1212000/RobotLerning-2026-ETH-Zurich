"""
Evaluation script for SAC on the SO100 position tracking task.
Supports quantitative evaluation and GUI playback.
"""

import sys
from pathlib import Path
import argparse
import time
import re
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from envs.so100_rl_env import SO100RLEnv
from exercises.ex4_sac import SACAgent
from exercises.ex4_sac_config import SAC_PARAMETERS
from rl.common import set_seed


def find_latest_checkpoint(log_root: Path) -> Path:
    """
    Find the checkpoint from the most recently modified SAC run directory,
    and within that run select the checkpoint with the largest iteration number.

    Expected structure:
        logs/sac/<run_name>/iter_<N>.pt
    """
    if not log_root.exists():
        raise FileNotFoundError(f"SAC log directory not found: {log_root}")

    run_dirs = [p for p in log_root.iterdir() if p.is_dir() and p.name != "eval"]
    if not run_dirs:
        raise FileNotFoundError(f"No SAC run directories found under: {log_root}")

    run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    def iter_num(path: Path) -> int:
        match = re.fullmatch(r"iter_(\d+)\.pt", path.name)
        return int(match.group(1)) if match else -1

    for run_dir in run_dirs:
        checkpoints = [
            p for p in run_dir.glob("iter_*.pt")
            if p.is_file() and iter_num(p) >= 0
        ]
        if checkpoints:
            checkpoints.sort(key=iter_num)
            return checkpoints[-1]

    raise FileNotFoundError(
        f"No SAC checkpoints found under: {log_root}\n"
        f"Expected files like: logs/sac/<run_name>/iter_<N>.pt"
    )


def evaluate_policy(env, agent, num_episodes, real_time=False):
    returns = []
    lengths = []
    tracking_errors = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_return = 0.0
        episode_length = 0
        episode_errors = []

        while not done:
            with torch.inference_mode():
                obs_tensor = torch.as_tensor(
                    obs, dtype=torch.float32, device=agent.device
                ).unsqueeze(0)
                action = agent.predict_action(obs_tensor)
                action = action.cpu().numpy().squeeze(0)
                next_obs, reward, terminated, truncated, info = env.step(action)

            obs = next_obs
            episode_return += reward
            episode_length += 1
            episode_errors.append(float(info.get("ee_tracking_error", 0.0)))
            done = terminated or truncated

            if real_time:
                time.sleep(env.ctrl_timestep)

        mean_error = float(np.mean(episode_errors)) if episode_errors else 0.0

        returns.append(float(episode_return))
        lengths.append(int(episode_length))
        tracking_errors.append(mean_error)

        print(
            f"Eval Episode {episode + 1:02d} | "
            f"Return: {episode_return:.3f} | "
            f"Length: {episode_length} | "
            f"Mean EE Error: {mean_error:.6f}"
        )

    return returns, lengths, tracking_errors


def summarize_metrics(returns, lengths, tracking_errors):
    returns_np = np.array(returns, dtype=np.float32)
    lengths_np = np.array(lengths, dtype=np.int32)
    errors_np = np.array(tracking_errors, dtype=np.float32)

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
        "mean_tracking_error": float(np.mean(errors_np)),
        "std_tracking_error": float(np.std(errors_np)),
        "min_tracking_error": float(np.min(errors_np)),
        "max_tracking_error": float(np.max(errors_np)),
        "returns": returns,
        "lengths": lengths,
        "tracking_errors": tracking_errors,
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate or play a trained SAC policy on the SO100 tracking task."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help=(
            "Path to the trained SAC checkpoint. "
            "If omitted, automatically loads the largest-iteration checkpoint "
            "from the most recently trained run under logs/sac/<run_name>/."
        ),
    )
    parser.add_argument(
        "--num_eval_episodes",
        type=int,
        default=20,
        help="Number of evaluation episodes.",
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Open a GUI window and play the learned policy.",
    )
    args = parser.parse_args()

    config = SAC_PARAMETERS
    seed = config["seed"]
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU name: {torch.cuda.get_device_name(0)}")

    log_dir = ROOT_DIR / "logs" / "sac"
    if args.model_path is None:
        model_path = find_latest_checkpoint(log_dir)
        print(f"Auto-selected latest checkpoint: {model_path}")
    else:
        model_path = Path(args.model_path).expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    xml_path = (ROOT_DIR / "assets" / "mujoco" / "so100_pos_ctrl.xml").resolve()
    render_mode = "human" if args.play else None

    env = SO100RLEnv(xml_path=xml_path, render_mode=render_mode)

    if args.play:
        print("Play mode enabled: opening GUI window...")

    agent = SACAgent(
        obs_dim=env.state_dim,
        act_dim=env.action_dim,
        hidden_sizes=config["hidden_sizes"],
        actor_lr=config["actor_lr"],
        critic_lr=config["critic_lr"],
        alpha_lr=config["alpha_lr"],
        gamma=config["gamma"],
        tau=config["tau"],
        init_alpha=config["init_alpha"],
        target_entropy=config["target_entropy"],
        device=device,
    )

    agent.load(str(model_path))
    agent.eval_mode()
    print(f"Loaded checkpoint from: {model_path}")

    try:
        returns, lengths, tracking_errors = evaluate_policy(
            env=env,
            agent=agent,
            num_episodes=args.num_eval_episodes,
            real_time=args.play,
        )
    except KeyboardInterrupt:
        print("\n[Eval] Interrupted by user, shutting down viewer cleanly...")
        env.close()
        sys.exit(0)

    env.close()

    metrics = summarize_metrics(
        returns=returns,
        lengths=lengths,
        tracking_errors=tracking_errors,
    )
    metrics["model_path"] = str(model_path)

    print("\n===== Evaluation Summary =====")
    print(f"Number of episodes   : {metrics['num_episodes']}")
    print(f"Mean return          : {metrics['mean_return']:.3f}")
    print(f"Std return           : {metrics['std_return']:.3f}")
    print(f"Min return           : {metrics['min_return']:.3f}")
    print(f"Max return           : {metrics['max_return']:.3f}")
    print(f"Median return        : {metrics['median_return']:.3f}")
    print(f"Mean length          : {metrics['mean_length']:.2f}")
    print(f"Std length           : {metrics['std_length']:.2f}")
    print(f"Mean tracking error  : {metrics['mean_tracking_error']:.6f}")
    print(f"Std tracking error   : {metrics['std_tracking_error']:.6f}")
    print(f"Min tracking error   : {metrics['min_tracking_error']:.6f}")
    print(f"Max tracking error   : {metrics['max_tracking_error']:.6f}")


if __name__ == "__main__":
    main()