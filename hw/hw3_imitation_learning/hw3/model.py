"""Model definitions for SO-100 imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        """Compute training loss for a batch."""
        raise NotImplementedError

    @abc.abstractmethod
    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""
        raise NotImplementedError


class ResidualBlock(nn.Module):
    """Pre-norm MLP block with a residual connection."""

    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ChunkTrunk(nn.Module):
    """Maps a state vector to a flat action chunk of size chunk_size * action_dim.

    A plain MLP is enough for this task, but the residual + pre-norm structure
    makes it trainable at larger depths without any learning-rate babysitting.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        d_model: int,
        depth: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim

        self.stem = nn.Sequential(
            nn.Linear(state_dim, d_model),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(d_model, dropout) for _ in range(depth)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, chunk_size * action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        h = self.blocks(self.stem(state))
        flat = self.head(h)
        return flat.reshape(-1, self.chunk_size, self.action_dim)


class ObstaclePolicy(BasePolicy):
    """Predicts action chunks with an MSE loss.

    A simple MLP that maps a state vector to a flat action chunk
    (chunk_size * action_dim) and reshapes to (B, chunk_size, action_dim).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int = 16,
        d_model: int = 384,
        depth: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        # Kept as attributes so a checkpoint can be reconstructed exactly.
        self.d_model = d_model
        self.depth = depth
        self.dropout = dropout
        self.net = ChunkTrunk(
            state_dim, action_dim, chunk_size, d_model, depth, dropout
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return predicted action chunk of shape (B, chunk_size, action_dim)."""
        return self.net(state)

    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        pred = self(state)
        return nn.functional.mse_loss(pred, action_chunk)

    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        return self(state)


class MultiTaskPolicy(BasePolicy):
    """Goal-conditioned policy for the multicube scene.

    Same regression objective as :class:`ObstaclePolicy`, but with more
    capacity and a little dropout: the goal one-hot and the randomised bin
    position are part of the state vector, so the network has to represent
    several distinct behaviours at once and overfits more easily.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int = 16,
        d_model: int = 512,
        depth: int = 4,
        dropout: float = 0.05,
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        self.d_model = d_model
        self.depth = depth
        self.dropout = dropout
        self.net = ChunkTrunk(
            state_dim, action_dim, chunk_size, d_model, depth, dropout
        )

    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        pred = self(state)
        return nn.functional.mse_loss(pred, action_chunk)

    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        return self(state)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return predicted action chunk of shape (B, chunk_size, action_dim)."""
        return self.net(state)


PolicyType: TypeAlias = Literal["obstacle", "multitask"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int = 16,
    d_model: int | None = None,
    depth: int | None = None,
    dropout: float | None = None,
) -> BasePolicy:
    # ``None`` means "use the policy's own default" so the two policy types can
    # keep different architecture defaults.
    kwargs: dict[str, object] = {}
    if d_model is not None:
        kwargs["d_model"] = d_model
    if depth is not None:
        kwargs["depth"] = depth
    if dropout is not None:
        kwargs["dropout"] = dropout

    if policy_type == "obstacle":
        return ObstaclePolicy(
            action_dim=action_dim,
            state_dim=state_dim,
            chunk_size=chunk_size,
            **kwargs,
        )
    if policy_type == "multitask":
        return MultiTaskPolicy(
            action_dim=action_dim,
            state_dim=state_dim,
            chunk_size=chunk_size,
            **kwargs,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
