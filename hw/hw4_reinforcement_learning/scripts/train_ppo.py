"""
Training script for PPO on the SO100 position tracking task.
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from hw.hw4_reinforcement_learning.envs.so100_rl_env import SO100RLEnv
from hw.hw4_reinforcement_learning.exercises.ex3_ppo import PPOAgent
from hw.hw4_reinforcement_learning.exercises.ex3_ppo_config import PPO_PARAMETERS
from hw.hw4_reinforcement_learning.rl.buffers import RolloutBuffer
from hw.hw4_reinforcement_learning.rl.common import ensure_dir, set_seed


def evaluate_policy(env, agent, num_episodes=5):
    returns = []
    lengths = []
    ee_tracking_errors = []

    agent.eval_mode()
    with torch.inference_mode():
        for _ in range(num_episodes):
            obs, _ = env.reset()
            done = False
            episode_return = 0.0
            episode_length = 0

            while not done:
                obs = torch.as_tensor(obs, dtype=torch.float, device=agent.device).unsqueeze(0)
                action = agent.predict_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action.cpu().numpy().squeeze(0))

                obs = next_obs
                episode_return += reward
                episode_length += 1
                done = terminated or truncated
            
            returns.append(float(episode_return))
            lengths.append(int(episode_length))
            ee_tracking_errors.append(info["ee_tracking_error"])

    return float(np.mean(returns)), float(np.mean(lengths)), float(np.mean(ee_tracking_errors))


def main():
    config = PPO_PARAMETERS

    seed = config["seed"]
    total_iterations = config["total_iterations"]
    n_steps = config["n_steps"]
    gamma = config["gamma"]
    gae_lambda = config["gae_lambda"]
    save_interval = config["save_interval"]

    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU name: {torch.cuda.get_device_name(0)}")

    xml_path = ROOT_DIR / "assets" / "mujoco" / "so100_pos_ctrl.xml"
    env = SO100RLEnv(xml_path=xml_path, render_mode=None)
    eval_env = SO100RLEnv(xml_path=xml_path, render_mode=None)

    agent = PPOAgent(
        obs_dim=env.state_dim,
        act_dim=env.action_dim,
        hidden_sizes=config["hidden_sizes"],
        n_steps=n_steps,
        mini_batch_size=config["mini_batch_size"],
        n_epochs=config["n_epochs"],
        gamma=gamma,
        gae_lambda=gae_lambda,
        surrogate_loss_coeff=config["surrogate_loss_coeff"],
        value_loss_coeff=config["value_loss_coeff"],
        entropy_coeff=config["entropy_coeff"],
        clip_ratio=config["clip_ratio"],
        learning_rate=config["learning_rate"],
        target_kl=config["target_kl"],
        max_grad_norm=config["max_grad_norm"],
        device=device,
    )

    buffer = RolloutBuffer(
        obs_dim=env.state_dim,
        act_dim=env.action_dim,
        size=n_steps,
        gamma=gamma,
        gae_lambda=gae_lambda,
        device=device,
    )

    log_dir = ROOT_DIR / "logs" / "ppo"
    run_name = datetime.now().strftime("%y_%m_%d_%H_%M_%S_model")
    run_dir = ensure_dir(log_dir / run_name)
    writer = SummaryWriter(log_dir=run_dir)

    obs, _ = env.reset()
    obs = torch.as_tensor(obs, dtype=torch.float, device=device).unsqueeze(0)
    done = False
    

    for it in range(total_iterations):
        agent.train_mode()
        with torch.inference_mode():
            for _ in range(n_steps):
                if done:
                    obs, _ = env.reset()
                    obs = torch.as_tensor(obs, dtype=torch.float, device=device).unsqueeze(0)

                action_raw, action_env, value, logp, mu, std = agent.select_action(obs)
                next_obs, reward, terminated, timeout, info = env.step(action_env.cpu().numpy().squeeze(0))

                next_obs = torch.as_tensor(next_obs, dtype=torch.float, device=device).unsqueeze(0)
                done = terminated or timeout

                if timeout:
                    # bootstrap value at the end of trajectory
                    reward += gamma * agent.critic(next_obs).item()

                buffer.store(
                    obs=obs,
                    act=action_raw,
                    rew=reward,
                    done=done,
                    val=value,
                    logp=logp,
                    mu=mu,
                    std=std
                )

                obs = next_obs

            last_val = agent.critic(obs).item()
            buffer.compute_returns(last_val)
            rollout_batch = buffer.get(device=device)

        stats = agent.update(rollout_batch)
        mean_eval_return, mean_eval_length, mean_ee_tracking_error = evaluate_policy(eval_env, agent, num_episodes=5)
        iteration = it + 1

        writer.add_scalar("train/step", n_steps * iteration, iteration)
        writer.add_scalar("train/learning_rate", agent.learning_rate, iteration)
        writer.add_scalar("train/global_action_std", agent.actor.action_std.mean().item(), iteration)
        writer.add_scalar("train/mean_kl", float(stats.mean_kl), iteration)
        writer.add_scalar("train/mean_surrogate_loss", stats.mean_surrogate_loss, iteration)
        writer.add_scalar("train/mean_value_loss", stats.mean_value_loss, iteration)
        writer.add_scalar("train/mean_entropy", stats.mean_entropy, iteration)
        writer.add_scalar("eval/return", mean_eval_return, iteration)
        writer.add_scalar("eval/length", mean_eval_length, iteration)
        writer.add_scalar("eval/ee_tracking_error", mean_ee_tracking_error, iteration)

        print(
            f"[PPO] iteration={iteration}/{total_iterations} "
            f"step={n_steps * iteration} "
            f"mean_ee_tracking_error={mean_ee_tracking_error:.4f} "
            f"learning_rate={agent.learning_rate:.2e} "
            f"global_std={agent.actor.action_std.mean().item():.4f} "
            f"mean_kl={stats.mean_kl:.4f} "
            f"mean_surrogate_loss={stats.mean_surrogate_loss:.4f} "
            f"mean_value_loss={stats.mean_value_loss:.4f} "
            f"mean_entropy={stats.mean_entropy:.4f} "
            f"eval_return={mean_eval_return:.4f} "
            f"eval_length={mean_eval_length:.2f}"
        )

        if (it + 1) % save_interval == 0 or (it + 1) == total_iterations:
            model_path = run_dir / f"iter_{it + 1}.pt"
            agent.save(model_path)

    env.close()
    eval_env.close()
    writer.close()


if __name__ == "__main__":
    main()