"""Global seeding for reproducible runs.

Every entry point seeds before doing anything else. Reproducibility is treated
as a feature here rather than a nicety: the project's claims rest on repeated
cross-validation with persisted folds, and a run that cannot be reproduced
cannot be compared against another.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch, including CUDA.

    Args:
        seed: Value applied to every generator.
        deterministic: Force deterministic cuDNN kernels. This costs throughput
            and is worth it for reported runs; set False for exploratory work
            where speed matters more than bit-exact reproduction.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info("seeded everything with %d (deterministic=%s)", seed, deterministic)
