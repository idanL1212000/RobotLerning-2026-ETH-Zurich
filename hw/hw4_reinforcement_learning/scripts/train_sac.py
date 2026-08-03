"""
Training script for SAC on the SO100 position tracking task.
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
from hw.hw4_reinforcement_learning.exercises.ex4_sac import SACAgent, SACUpdateStats
from hw.hw4_reinforcement_learning.exercises.ex4_sac_config import SAC_PARAMETERS
from hw.hw4_reinforcement_learning.rl.buffers import ReplayBuffer
from hw.hw4_reinforcement_learning.rl.common import ensure_dir, set_seed


def evaluate_policy(env: SO100RLEnv, agent: SACAgent, num_episodes=5):
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
    config = SAC_PARAMETERS

    seed = config["seed"]
    total_iterations = config["total_iterations"]
    learning_start_steps = config["learning_start_steps"]
    train_freq = config["train_freq"]
    gradient_steps = config["gradient_steps"]
    batch_size = config["batch_size"]
    eval_freq = config["eval_freq"]
    replay_size = config["replay_size"]
    save_interval = config["save_interval"]

    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU name: {torch.cuda.get_device_name(0)}")

    xml_path = ROOT_DIR / "assets" / "mujoco" / "so100_pos_ctrl.xml"
    env = SO100RLEnv(xml_path=xml_path, render_mode=None)
    eval_env = SO100RLEnv(xml_path=xml_path, render_mode=None)

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

    replay_buffer = ReplayBuffer(
        obs_dim=env.state_dim,
        act_dim=env.action_dim,
        max_size=replay_size,
        device=device,
    )

    log_dir = ROOT_DIR / "logs" / "sac"
    run_name = datetime.now().strftime("%y_%m_%d_%H_%M_%S_model")
    run_dir = ensure_dir(log_dir / run_name)
    writer = SummaryWriter(log_dir=run_dir)

    obs, _ = env.reset()
    obs = torch.as_tensor(obs, dtype=torch.float, device=device).unsqueeze(0)

    it = 0
    step = 0
    eval_step = 0
    while it < total_iterations:
        agent.train_mode()
        with torch.no_grad():
            step += 1
            if step < learning_start_steps:
                action = torch.empty(env.action_dim, dtype=torch.float, device=device).uniform_(-1.0, 1.0).unsqueeze(0)
            else:
                action = agent.sample_action(obs)

            next_obs, reward, terminated, truncated, info = env.step(action.cpu().numpy().squeeze(0))
            next_obs = torch.as_tensor(next_obs, dtype=torch.float, device=device).unsqueeze(0)
            done = terminated or truncated

            replay_buffer.store(
                obs=obs.squeeze(0),  # shape [obs_dim]
                act=action.squeeze(0),  # shape [action_dim]
                rew=reward,
                next_obs=next_obs.squeeze(0),  # shape [obs_dim]
                done=done,
            )

            obs = next_obs

            if done:
                obs, _ = env.reset()
                obs = torch.as_tensor(obs, dtype=torch.float, device=device).unsqueeze(0)

        if step >= learning_start_steps and step % train_freq == 0:
            mean_stats = SACUpdateStats.init_lists()

            for _ in range(gradient_steps):
                batch = replay_buffer.sample_batch(batch_size=batch_size)
                stats = agent.update(batch)
                mean_stats.append(stats)

            mean_stats = mean_stats.mean()
            it += 1

        if step % eval_freq == 0:
            eval_step += 1

            mean_eval_return, mean_eval_length, mean_eval_ee_tracking_error = evaluate_policy(
                eval_env, agent, num_episodes=5
            )

            writer.add_scalar("train/step", step, eval_step)
            writer.add_scalar("train/critic_loss", mean_stats.critic_loss, eval_step)
            writer.add_scalar("train/actor_loss", mean_stats.actor_loss, eval_step)
            writer.add_scalar("train/alpha_loss", mean_stats.alpha_loss, eval_step)
            writer.add_scalar("train/alpha", mean_stats.alpha, eval_step)
            writer.add_scalar("eval/return", mean_eval_return, eval_step)
            writer.add_scalar("eval/length", mean_eval_length, eval_step)
            writer.add_scalar("eval/ee_tracking_error", mean_eval_ee_tracking_error, eval_step)

            print(
                f"[SAC] eval_step={eval_step} "
                f"iteration={it}/{total_iterations} "
                f"step={step} "
                f"ee_tracking_error={mean_eval_ee_tracking_error:.4f} "
                f"critic_loss={mean_stats.critic_loss:.4f} "
                f"actor_loss={mean_stats.actor_loss:.4f} "
                f"alpha_loss={mean_stats.alpha_loss:.4f} "
                f"alpha={mean_stats.alpha:.4f} "
                f"eval_return={mean_eval_return:.4f} "
                f"eval_length={mean_eval_length:.2f}"
            )

            if eval_step % save_interval == 0:
                model_path = run_dir / f"iter_{it}.pt"
                agent.save(model_path)

    final_model_path = run_dir / f"iter_{it}.pt"
    agent.save(final_model_path)

    env.close()
    eval_env.close()
    writer.close()


if __name__ == "__main__":
    main()