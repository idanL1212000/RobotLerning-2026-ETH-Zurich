"""
Training script for DQN on CartPole-v1.
"""

import sys
from pathlib import Path
import random

import numpy as np
import torch
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from hw.hw4_reinforcement_learning.envs.cartpole_wrapper import CartPoleWrapper
from hw.hw4_reinforcement_learning.exercises.ex2_dqn import ReplayBuffer, DQN
from hw.hw4_reinforcement_learning.exercises.ex2_dqn_config import DQN_PARAMETERS


def train_off_policy_agent(env, agent, num_episodes, replay_buffer,
                           minimal_size, batch_size):
    """
    Train an off-policy agent with a replay buffer.
    """
    return_list = []

    for i in range(10):
        print(f"Iteration {i}")

        for i_episode in range(int(num_episodes / 10)):
            episode_return = 0.0
            state = env.reset()
            done = False

            while not done:
                action = agent.take_action(state)
                next_state, reward, done, _ = env.step(action)

                replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                episode_return += reward

                if replay_buffer.size() > minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                    transition_dict = {
                        "states": b_s,
                        "actions": b_a,
                        "rewards": b_r,
                        "next_states": b_ns,
                        "dones": b_d,
                    }
                    agent.update(transition_dict)

            return_list.append(episode_return)

            if (i_episode + 1) % 10 == 0:
                episode_id = int(num_episodes / 10) * i + i_episode + 1
                mean_return = np.mean(return_list[-10:])
                print(f"Episode {episode_id}, Average Return: {mean_return:.3f}")

    return return_list


def main():
    # Hyperparameters
    lr = DQN_PARAMETERS["lr"]
    epsilon = DQN_PARAMETERS["epsilon"]
    target_update = DQN_PARAMETERS["target_update"]
    hidden_dim = DQN_PARAMETERS["hidden_dim"]

    gamma = DQN_PARAMETERS["gamma"]
    num_episodes = DQN_PARAMETERS["num_episodes"]
    buffer_size = DQN_PARAMETERS["buffer_size"]
    minimal_size = DQN_PARAMETERS["minimal_size"]
    batch_size = DQN_PARAMETERS["batch_size"]
    seed = DQN_PARAMETERS["seed"]

    # Random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU name: {torch.cuda.get_device_name(0)}")

    # Environment
    env = CartPoleWrapper(seed=seed)

    # Replay buffer
    replay_buffer = ReplayBuffer(buffer_size)

    # Agent
    agent = DQN(
        state_dim=env.state_dim,
        hidden_dim=hidden_dim,
        action_dim=env.action_dim,
        learning_rate=lr,
        gamma=gamma,
        epsilon=epsilon,
        target_update=target_update,
        device=device,
    )

    # Train
    return_list = train_off_policy_agent(
        env=env,
        agent=agent,
        num_episodes=num_episodes,
        replay_buffer=replay_buffer,
        minimal_size=minimal_size,
        batch_size=batch_size,
    )

    env.close()

    # Logging
    log_dir = ROOT_DIR / "logs" / "dqn"
    model_dir = log_dir / "models"
    result_dir = log_dir / "results"

    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = model_dir / "dqn_cartpole.pth"
    agent.save(model_path)
    print(f"Model saved to: {model_path}")

    # Plot raw training curve
    episodes_list = list(range(len(return_list)))
    plt.figure()
    plt.plot(episodes_list, return_list)
    plt.xlabel("Episodes")
    plt.ylabel("Returns")
    plt.title("DQN on CartPole-v1")

    train_curve_path = result_dir / "dqn_training_curve.png"
    plt.savefig(train_curve_path, bbox_inches="tight")
    plt.close()
    print(f"Training curve saved to: {train_curve_path}")


if __name__ == "__main__":
    main()