from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from itertools import chain
from collections.abc import Generator

from hw.hw4_reinforcement_learning.rl.networks import GaussianActor, ValueNet
from hw.hw4_reinforcement_learning.rl.buffers import RolloutBatch


@dataclass
class PPOUpdateStats:
    mean_kl: float
    mean_surrogate_loss: float
    mean_value_loss: float
    mean_entropy: float


class PPOAgent:
    """
    Proximal Policy Optimization (PPO) agent for continuous action spaces.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes,
        n_steps: int = 2048,
        mini_batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        surrogate_loss_coeff: float = 1.0,
        value_loss_coeff: float = 1.0,
        entropy_coeff: float = 0.0,
        clip_ratio: float = 0.2,
        learning_rate: float = 1.0e-3,
        target_kl: float = 0.05,
        max_grad_norm: float = 1.0,
        device: torch.device = torch.device("cpu"),
    ):
        self.n_steps = n_steps
        self.mini_batch_size = mini_batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.surrogate_loss_coeff = surrogate_loss_coeff
        self.value_loss_coeff = value_loss_coeff
        self.entropy_coeff = entropy_coeff
        self.clip_ratio = clip_ratio
        self.learning_rate = learning_rate
        self.target_kl = target_kl
        self.max_grad_norm = max_grad_norm
        self.device = device

        self.actor = GaussianActor(obs_dim, act_dim, hidden_sizes).to(self.device)
        self.critic = ValueNet(obs_dim, hidden_sizes).to(self.device)

        # Combined optimizer for actor and critic
        self.optimizer = optim.Adam(
            chain(self.actor.parameters(), self.critic.parameters()),
            lr=self.learning_rate,
        )

    def select_action(self, obs: torch.Tensor):
        """
        Sample an action from the current policy.

        Args:
            obs (torch.Tensor): observation tensor

        Returns:
            action (torch.Tensor): sampled action
            action_clipped (torch.Tensor): action clipped into [-1, 1]
            value (float): critic prediction V(s)
            action_log_prob (float): log pi(a|s)
            action_mu (torch.Tensor): mean of Gaussian policy
            action_std (torch.Tensor): std of Gaussian policy
        """
        with torch.inference_mode():
            # Sample the raw (unclipped) action: this is the action whose
            # log-probability is consistent with the Gaussian policy and is
            # therefore the one that has to be stored for the PPO ratio.
            action = self.actor.act(obs)
            # Only the environment sees the clipped action.
            action_clipped = torch.clamp(action, -1.0, 1.0)
            action_log_prob = self.actor.get_actions_log_prob(action)
            action_mu = self.actor.action_mean
            action_std = self.actor.action_std
            value = self.critic(obs)

        return action, action_clipped, value, action_log_prob, action_mu, action_std

    def predict_action(self, obs: torch.Tensor):
        """
        Deterministic action for evaluation.
        """
        action = self.actor.act_inference(obs)
        return torch.clamp(action, -1.0, 1.0)

    def compute_kl_mean(self, old_mu_batch, old_std_batch, mu_batch, std_batch):
        """
        Compute the mean KL divergence between two Gaussian action distributions.

        Args:
            old_mu_batch (torch.Tensor): old policy mean
            old_std_batch (torch.Tensor): old policy std
            mu_batch (torch.Tensor): new policy mean
            std_batch (torch.Tensor): new policy std

        Returns:
            torch.Tensor: scalar mean KL divergence
        """
        # Closed-form KL(old || new) for diagonal Gaussians, per dimension.
        kl_per_dim = (
            torch.log(std_batch / (old_std_batch + 1e-8) + 1e-8)
            + (old_std_batch.pow(2) + (old_mu_batch - mu_batch).pow(2))
            / (2.0 * std_batch.pow(2))
            - 0.5
        )
        # Independent dimensions -> the joint KL is the sum over dimensions.
        kl_per_sample = kl_per_dim.sum(dim=-1)

        return kl_per_sample.mean()
        

    def adjust_learning_rate(self, kl, current_lr, min_lr=1e-5, max_lr=1e-3):
        """
        Adjust learning rate according to KL divergence.
        """
        new_lr = current_lr
        if kl > self.target_kl * 2.0:
            new_lr = max(current_lr / 1.5, min_lr)
        elif kl < self.target_kl / 1.5 and kl > 0:
            new_lr = min(current_lr * 1.5, max_lr)
        return new_lr

    def compute_surrogate_loss(self, logp_batch, old_logp_batch, adv_batch):
        """
        Compute the PPO clipped surrogate loss.

        Args:
            logp_batch (torch.Tensor): new log probabilities
            old_logp_batch (torch.Tensor): old log probabilities
            adv_batch (torch.Tensor): advantage estimates

        Returns:
            torch.Tensor: scaled surrogate loss
        """
        # Importance sampling ratio pi_new(a|s) / pi_old(a|s).
        ratio = torch.exp(logp_batch - old_logp_batch)
        clipped_ratio = torch.clamp(
            ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio
        )
        # Pessimistic bound on the objective; negated because we minimize.
        surrogate_loss = -torch.min(
            ratio * adv_batch, clipped_ratio * adv_batch
        ).mean()

        return self.surrogate_loss_coeff * surrogate_loss

    def compute_value_loss(self, val_batch, old_val_batch, ret_batch):
        """
        Compute value loss with clipping.
        """
        value_loss_unclipped = (val_batch - ret_batch).pow(2)
        # Keep the new value prediction within a trust region of the old one.
        value_clipped = old_val_batch + torch.clamp(
            val_batch - old_val_batch, -self.clip_ratio, self.clip_ratio
        )
        value_loss_clipped = (value_clipped - ret_batch).pow(2)
        # Pessimistic (larger) of the two losses.
        value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()

        return self.value_loss_coeff * value_loss

    def compute_entropy_loss(self, entropy_batch):
        """
        Compute entropy regularization term.
        """
        # Maximizing entropy == minimizing its negative.
        return -self.entropy_coeff * entropy_batch.mean()

    def mini_batch_generator(self, batch) -> Generator:
        """
        Generate mini-batches of data for PPO update.
        """
        for _ in range(self.n_epochs):
            indices = torch.randperm(
                self.n_steps, requires_grad=False, device=self.device
            )
            for start in range(0, self.n_steps, self.mini_batch_size):
                end = start + self.mini_batch_size
                batch_indices = indices[start:end]
                yield RolloutBatch(
                    obs=batch.obs[batch_indices],
                    act=batch.act[batch_indices],
                    logp=batch.logp[batch_indices],
                    mu=batch.mu[batch_indices],
                    std=batch.std[batch_indices],
                    val=batch.val[batch_indices],
                    ret=batch.ret[batch_indices],
                    adv=batch.adv[batch_indices],
                )

    def update(self, rollout_batch) -> PPOUpdateStats:
        """
        Update PPO actor and critic using a full rollout batch.

        Args:
            rollout_batch: a full batch collected from environment interaction

        Returns:
            PPOUpdateStats: statistics averaged over all mini-batch updates
        """
        mean_kl = 0
        mean_surrogate_loss = 0
        mean_value_loss = 0
        mean_entropy = 0
        num_updates = 0

        for mini_batch in self.mini_batch_generator(rollout_batch):
            obs_batch = mini_batch.obs
            act_batch = mini_batch.act
            old_logp_batch = mini_batch.logp
            old_mu_batch = mini_batch.mu
            old_std_batch = mini_batch.std
            old_val_batch = mini_batch.val
            ret_batch = mini_batch.ret
            adv_batch = mini_batch.adv

            self.actor.update_distribution(obs_batch)
            logp_batch = self.actor.get_actions_log_prob(act_batch)
            mu_batch = self.actor.action_mean
            std_batch = self.actor.action_std
            val_batch = self.critic(obs_batch)
            entropy_batch = self.actor.entropy

            # KL is only a diagnostic used to schedule the learning rate, so it
            # must not contribute gradients.
            with torch.no_grad():
                kl = self.compute_kl_mean(
                    old_mu_batch, old_std_batch, mu_batch, std_batch
                ).item()

            self.learning_rate = self.adjust_learning_rate(kl, self.learning_rate)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate

            surrogate_loss = self.compute_surrogate_loss(
                logp_batch, old_logp_batch, adv_batch
            )
            value_loss = self.compute_value_loss(val_batch, old_val_batch, ret_batch)
            entropy_loss = self.compute_entropy_loss(entropy_batch)
            loss = surrogate_loss + value_loss + entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(chain(self.actor.parameters(), self.critic.parameters()), self.max_grad_norm)
            self.optimizer.step()

            mean_kl += kl
            mean_surrogate_loss += surrogate_loss.item()
            mean_value_loss += value_loss.item()
            mean_entropy += entropy_batch.mean().item()
            num_updates += 1

        mean_kl /= num_updates
        mean_surrogate_loss /= num_updates
        mean_value_loss /= num_updates
        mean_entropy /= num_updates

        return PPOUpdateStats(
            mean_kl=mean_kl,
            mean_surrogate_loss=mean_surrogate_loss,
            mean_value_loss=mean_value_loss,
            mean_entropy=mean_entropy,
        )

    def save(self, path) -> None:
        """
        Save model parameters and optimizer state.
        """
        checkpoint = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, path)

    def load(self, path) -> None:
        """
        Load model parameters and optimizer state.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

    def train_mode(self) -> None:
        """
        Set actor and critic to training mode.
        """
        self.actor.train()
        self.critic.train()

    def eval_mode(self) -> None:
        """
        Set actor and critic to evaluation mode.
        """
        self.actor.eval()
        self.critic.eval()