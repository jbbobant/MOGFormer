"""Pretrain the self-supervised encoder on a cohort bundle, one run per seed.

Every run of one configuration shares its patient splits and its frozen gene
axis (both fixed by ``train.split_seed``), so the seeds differ only in the
model's initialisation, mask draws and batch order: the spread across them is
the seed noise against which a change to the encoder is judged.

Each run writes a self-contained directory under
``ssl_pretraining/<configuration name>_seed<seed>/``: the frozen encoder, its
graph tensors and inputs, every patient read-out, held-out reconstruction,
the training history and a provenance record.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from mogformer.analysis.embedding import run_pretraining
from mogformer.config import ExperimentConfig, load_config
from mogformer.evaluation.runner import load_cohort
from mogformer.experiments.paths import BundleLayout, provenance_record, rebase_config
from mogformer.training.losses import participation_ratio

logger = logging.getLogger(__name__)


def layout_for(config: ExperimentConfig, root: Path) -> BundleLayout:
    """Describe where a self-supervised configuration's inputs sit in a bundle.

    Args:
        config: Configuration as written in ``configs/ssl/``.
        root: The bundle directory.

    Returns:
        The layout, with the configuration's cohort archive and edge list.

    Raises:
        ValueError: If the configuration names no cohort archive.
    """
    if config.data.cohort_file is None:
        raise ValueError("a self-supervised configuration must set data.cohort_file")
    edges = Path(config.graph.edges_file or "string_edges.tsv").name
    return BundleLayout(root, cohort_file=config.data.cohort_file, edges_file=edges)


def run_ssl_pretraining(
    config_path: Path | str,
    bundle: Path | str,
    output_dir: Path | str,
    seeds: Sequence[int] = (42,),
    device: str | None = None,
    max_epochs: int | None = None,
) -> dict[str, Any]:
    """Pretrain one configuration once per seed.

    Args:
        config_path: Self-supervised configuration file.
        bundle: The cohort bundle directory.
        output_dir: Root the runs are written under.
        seeds: Training seeds, one run each.
        device: Torch device; CUDA when available if None.
        max_epochs: Cap on training epochs, for timing and smoke tests only.
            A capped run is renamed ``..._smoke_<n>ep`` so it cannot be taken
            for a result.

    Returns:
        Per run: directory, epochs, minutes, best monitoring loss and the
        participation ratio of each read-out.
    """
    raw = load_config(config_path)
    layout = layout_for(raw, Path(bundle)).require(with_folds=False)
    base = rebase_config(raw, layout, output_dir)
    if max_epochs is not None:
        base.train.max_epochs = max_epochs
        base.train.patience = min(base.train.patience, max_epochs)
        logger.warning("TRUNCATED RUN: %d epochs only", max_epochs)

    data = load_cohort(base)
    torch_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    written: dict[str, Any] = {}
    for seed in seeds:
        suffix = f"_seed{seed}" + (
            f"__smoke_{max_epochs}ep" if max_epochs is not None else ""
        )
        config = replace(
            base,
            name=f"{base.name}{suffix}",
            train=replace(base.train, seed=seed),
        )
        bundle_out = run_pretraining(
            config,
            data,
            device=torch_device,
            provenance=provenance_record({"configuration_file": str(config_path)}),
        )
        written[config.name] = {
            "output_dir": str(config.output_dir),
            "epochs": bundle_out.record["epochs_run"],
            "minutes": bundle_out.record["minutes"],
            "seconds_per_epoch": bundle_out.record["seconds_per_epoch"],
            "best_monitor_loss": bundle_out.best_val,
            "participation_ratio": {
                name: round(participation_ratio(values), 2)
                for name, values in bundle_out.readouts.items()
            },
        }
    return written


__all__ = ["layout_for", "run_ssl_pretraining"]
