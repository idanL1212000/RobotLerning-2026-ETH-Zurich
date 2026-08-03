from dataclasses import dataclass
import torch


@dataclass
class ReplayBatch:
    obs: torch.Tensor
    act: torch.Tensor
    rew: torch.Tensor
    next_obs: torch.Tensor
    done: torch.Tensor


class ReplayBuffer:
    """
    A simple FIFO replay buffer for off-policy algorithms such as SAC.
    """

    def __init__(self, obs_dim: int, act_dim: int, max_size: int, device: torch.device):
        self.obs_buf = torch.zeros((max_size, obs_dim), dtype=torch.float, device=device)
        self.next_obs_buf = torch.zeros((max_size, obs_dim), dtype=torch.float, device=device)
        self.act_buf = torch.zeros((max_size, act_dim), dtype=torch.float, device=device)
        self.rew_buf = torch.zeros(max_size, dtype=torch.float, device=device)
        self.done_buf = torch.zeros(max_size, dtype=torch.float, device=device)

        self.max_size = max_size
        self.ptr = 0
        self.size = 0

    def store(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        rew: float,
        next_obs: torch.Tensor,
        done: bool,
    ) -> None:
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size: int) -> ReplayBatch:
        idxs = torch.randint(0, self.size, size=[batch_size])

        return ReplayBatch(
            obs=self.obs_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            done=self.done_buf[idxs],
        )


@dataclass
class RolloutBatch:
    obs: torch.Tensor
    act: torch.Tensor
    logp: torch.Tensor
    mu: torch.Tensor
    std: torch.Tensor
    val: torch.Tensor
    ret: torch.Tensor
    adv: torch.Tensor


class RolloutBuffer:
    """
    Rollout buffer for on-policy algorithms such as PPO.
    Stores one rollout of fixed length.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        size: int,
        gamma: float,
        gae_lambda: float,
        device: torch.device,
    ):
        self.obs_buf = torch.zeros((size, obs_dim), dtype=torch.float, device=device)
        self.act_buf = torch.zeros((size, act_dim), dtype=torch.float, device=device)
        self.rew_buf = torch.zeros(size, dtype=torch.float, device=device)
        self.done_buf = torch.zeros(size, dtype=torch.bool, device=device)
        self.val_buf = torch.zeros(size, dtype=torch.float, device=device)
        self.logp_buf = torch.zeros(size, dtype=torch.float, device=device)
        self.mu_buf = torch.zeros((size, act_dim), dtype=torch.float, device=device)
        self.std_buf = torch.zeros((size, act_dim), dtype=torch.float, device=device)

        self.adv_buf = torch.zeros(size, dtype=torch.float, device=device)
        self.ret_buf = torch.zeros(size, dtype=torch.float, device=device)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.max_size = size

        self.ptr = 0

    def store(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        rew: float,
        done: bool,
        val: float,
        logp: float,
        mu: torch.Tensor,
        std: torch.Tensor,
    ) -> None:
        if self.ptr >= self.max_size:
            raise ValueError("RolloutBuffer is full. Call get() before storing more data.")

        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.mu_buf[self.ptr] = mu
        self.std_buf[self.ptr] = std
        self.ptr += 1

    def compute_returns(self, last_val: float) -> None:
        """
        Compute GAE-Lambda advantages and rewards-to-go returns for the current trajectory.
        Should be called at the end of a rollout or when an episode ends.
        """
        advantage = 0
        for step in reversed(range(self.max_size)):
            if step == self.max_size - 1:
                next_val = last_val
            else:
                next_val = self.val_buf[step + 1]
            not_terminal = 1.0 - self.done_buf[step].float()
            delta = self.rew_buf[step] + self.gamma * next_val * not_terminal - self.val_buf[step]
            advantage = delta + self.gamma * self.gae_lambda * advantage * not_terminal
            self.adv_buf[step] = advantage
            self.ret_buf[step] = advantage + self.val_buf[step]

        self.adv_buf = (self.adv_buf - self.adv_buf.mean()) / (self.adv_buf.std() + 1e-8)

    def get(self, device: torch.device) -> RolloutBatch:
        """
        Get all data from the buffer. Buffer must be full.
        Advantages are normalized here.
        """
        if self.ptr != self.max_size:
            raise ValueError(
                f"RolloutBuffer must be full before calling get(). "
                f"Current size: {self.ptr}, expected: {self.max_size}"
            )

        batch = RolloutBatch(
            obs=self.obs_buf,
            act=self.act_buf,
            logp=self.logp_buf,
            mu=self.mu_buf,
            std=self.std_buf,
            val=self.val_buf,
            ret=self.ret_buf,
            adv=self.adv_buf,
        )
        # Reset buffer
        self.ptr = 0

        return batch