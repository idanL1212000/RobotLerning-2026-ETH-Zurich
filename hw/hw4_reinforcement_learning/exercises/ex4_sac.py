from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from hw.hw4_reinforcement_learning.rl.networks import DoubleQNet, SquashedGaussianActor
from hw.hw4_reinforcement_learning.rl.buffers import ReplayBatch


@dataclass
class SACUpdateStats:
    actor_loss: float | list[float]
    critic_loss: float | list[float]
    alpha_loss: float | list[float]
    alpha: float | list[float]

    @staticmethod
    def init_lists():
        return SACUpdateStats(
            actor_loss=[],
            critic_loss=[],
            alpha_loss=[],
            alpha=[],
        )

    def append(self, other) -> None:
        self.actor_loss.append(other.actor_loss)
        self.critic_loss.append(other.critic_loss)
        self.alpha_loss.append(other.alpha_loss)
        self.alpha.append(other.alpha)

    def mean(self):
        return SACUpdateStats(
            actor_loss=float(np.mean(self.actor_loss)),
            critic_loss=float(np.mean(self.critic_loss)),
            alpha_loss=float(np.mean(self.alpha_loss)),
            alpha=float(np.mean(self.alpha)),
        )


class SACAgent:
    """
    Soft Actor-Critic (SAC) agent for continuous control.

    Main components:
      - a squashed Gaussian policy actor
      - two Q-networks (double Q-learning)
      - two target Q-networks
      - entropy regularization with automatic temperature tuning
    
    SAC optimizes:
      
    Critic target:
        y = r + gamma * (1 - done) *
            [ min(Q1_target(s', a'), Q2_target(s', a')) - alpha * log pi(a'|s') ]

    Actor objective:
        J_pi = E[ alpha * log pi(a|s) - min(Q1(s, a), Q2(s, a)) ]

    Temperature objective:
        J_alpha = E[ -log_alpha * (log pi(a|s) + target_entropy) ]
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes,
        actor_lr: float,
        critic_lr: float,
        alpha_lr: float,
        gamma: float,
        tau: float,
        init_alpha: float,
        target_entropy,
        device: torch.device,
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.act_dim = act_dim

        self.actor = SquashedGaussianActor(obs_dim, act_dim, hidden_sizes).to(
            self.device
        )
        self.critic = DoubleQNet(obs_dim, act_dim, hidden_sizes).to(self.device)
        self.critic_target = DoubleQNet(obs_dim, act_dim, hidden_sizes).to(
            self.device
        )

        # Copy weights to target critics initially
        self.critic_target.load_state_dict(self.critic.state_dict())

        for p in self.critic_target.q1.parameters():
            p.requires_grad = False
        for p in self.critic_target.q2.parameters():
            p.requires_grad = False

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        if target_entropy is None:
            target_entropy = -float(act_dim)
        self.target_entropy = target_entropy

        self.log_alpha = torch.tensor(
            np.log(init_alpha),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

    @property
    def alpha(self) -> torch.Tensor:
        """
        Temperature parameter alpha = exp(log_alpha).
        """
        return self.log_alpha.exp()

    def sample_action(self, obs: torch.Tensor):
        """
        Sample an action for environment interaction.

        Returns:
            action (torch.Tensor): action within [-1, 1] due to tanh squashing
        """
        with torch.no_grad():
            action, _ = self.actor.act(obs)

        return action

    def predict_action(self, obs: torch.Tensor):
        """
        Deterministic action for evaluation.
        """
        return self.actor.act_inference(obs)

    def compute_critic_loss(self, obs, act, rew, next_obs, done) -> torch.Tensor:
        """
        Compute the SAC critic loss.

        Critic target:
            y = r + gamma * (1 - done) *
                [ min(Q1_target(s', a'), Q2_target(s', a')) - alpha * log pi(a'|s') ]

        Args:
            obs (torch.Tensor): current observations
            act (torch.Tensor): actions taken
            rew (torch.Tensor): rewards
            next_obs (torch.Tensor): next observations
            done (torch.Tensor): done flags

        Returns:
            torch.Tensor: critic loss
        """
        with torch.no_grad():
            # Next action comes from the *current* policy (off-policy correction).
            next_action, next_logp = self.actor.act(next_obs)
            q1_next, q2_next = self.critic_target(next_obs, next_action)
            # Clipped double-Q + entropy bonus of the soft value function.
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_logp
            target_q = rew + self.gamma * (1.0 - done) * q_next

        q1_pred, q2_pred = self.critic(obs, act)
        critic_loss = 0.5 * (
            F.mse_loss(q1_pred, target_q) + F.mse_loss(q2_pred, target_q)
        )

        return critic_loss

    def compute_actor_loss(self, obs, act_new, logp_new) -> torch.Tensor:
        """
        Compute the SAC actor loss.

        Actor objective:
            J_pi = E[ alpha * log pi(a|s) - min(Q1(s, a), Q2(s, a)) ]

        Args:
            obs (torch.Tensor): current observations
            act_new (torch.Tensor): newly sampled actions from current policy
            logp_new (torch.Tensor): log probabilities of sampled actions

        Returns:
            torch.Tensor: actor loss
        """
        q1_new, q2_new = self.critic(obs, act_new)
        q_new = torch.min(q1_new, q2_new)
        # alpha is optimized separately, so it is a constant here.
        actor_loss = (self.alpha.detach() * logp_new - q_new).mean()

        return actor_loss

    def compute_alpha_loss(self, logp_new) -> torch.Tensor:
        """
        Compute the SAC temperature loss.

        Temperature objective:
            J_alpha = E[ -log_alpha * (log pi(a|s) + target_entropy) ]

        Args:
            logp_new (torch.Tensor): log probabilities of newly sampled actions

        Returns:
            torch.Tensor: alpha loss
        """
        # Detached target: gradients flow into log_alpha only.
        # Entropy above target (logp below -target) -> alpha decreases, and
        # vice versa.
        alpha_loss = -(
            self.log_alpha * (logp_new + self.target_entropy).detach()
        ).mean()

        return alpha_loss

    def soft_update_targets(self) -> None:
        """
        Polyak averaging for the target network:
            target <- tau * online + (1 - tau) * target
        """
        with torch.no_grad():
            for target_param, param in zip(
                self.critic_target.parameters(), self.critic.parameters()
            ):
                target_param.data.copy_(
                    (1.0 - self.tau) * target_param.data + self.tau * param.data
                )

    def update(self, batch: ReplayBatch) -> SACUpdateStats:
        """
        One SAC update step:
          1. update critic
          2. update actor
          3. update alpha
          4. soft-update target critics

        Args:
            batch (ReplayBatch): mini-batch sampled from replay buffer

        Returns:
            SACUpdateStats: statistics of this update step
        """
        obs = batch.obs
        act = batch.act
        rew = batch.rew
        next_obs = batch.next_obs
        done = batch.done

        critic_loss = self.compute_critic_loss(obs, act, rew, next_obs, done)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Fresh on-policy samples for the actor and temperature objectives.
        act_new, logp_new = self.actor.act(obs)

        actor_loss = self.compute_actor_loss(obs, act_new, logp_new)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = self.compute_alpha_loss(logp_new)
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.soft_update_targets()

        return SACUpdateStats(
            actor_loss=actor_loss.item(),
            critic_loss=critic_loss.item(),
            alpha_loss=alpha_loss.item(),
            alpha=self.alpha.item(),
        )

    def save(self, path) -> None:
        """
        Save model parameters and optimizer states.
        """
        checkpoint = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
        }
        torch.save(checkpoint, path)

    def load(self, path) -> None:
        """
        Load model parameters and optimizer states.
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])

        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

        self.log_alpha.data.copy_(checkpoint["log_alpha"].to(self.device))
        self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])

    def train_mode(self) -> None:
        """
        Set modules to training mode.
        """
        self.actor.train()
        self.critic.train()
        self.critic_target.train()
        self.log_alpha.requires_grad_(True)

    def eval_mode(self) -> None:
        """
        Set modules to evaluation mode.
        """
        self.actor.eval()
        self.critic.eval()
        self.critic_target.eval()
        self.log_alpha.requires_grad_(False)