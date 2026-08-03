import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set random seed for Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path) -> Path:
    """
    Create directory if it does not exist and return it as a Path object.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path