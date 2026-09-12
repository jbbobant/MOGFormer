"""Command-line entry points, one per phase of the project.

Every subcommand takes a configuration file and writes its resolved settings
beside its outputs, so a result can always be traced back to what produced it.

Run ``mogformer --help`` for the available phases.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from mogformer.config import ExperimentConfig, load_config
from mogformer.data.folds import make_folds, save_folds
from mogformer.data.omics import load_omics
from mogformer.training.seed import seed_everything

logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    """Send logs to stderr at the requested level.

    Args:
        verbose: Emit debug-level records as well as info.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


def _load(config_path: str) -> ExperimentConfig:
    """Load a configuration and seed the run from it.

    Args:
        config_path: Path to the YAML configuration.

    Returns:
        The resolved configuration.
    """
    config = load_config(config_path)
    seed_everything(config.train.seed)
    return config


def command_folds(args: argparse.Namespace) -> int:
    """Build and persist the shared cross-validation partition.

    This runs first and once. Every model afterwards reads the file it writes,
    which is what makes the paired statistics valid.

    Args:
        args: Parsed arguments carrying ``config``.

    Returns:
        Process exit status.
    """
    config = _load(args.config)
    data = load_omics(
        raw_dir=config.data.raw_dir,
        rna_file=config.data.rna_file,
        cnv_file=config.data.cnv_file,
        methy_file=config.data.methy_file,
        clin_file=config.data.clin_file,
        label_col=config.data.label_col,
        exclude=config.data.exclude_classes,
    )
    folds = make_folds(
        data.y,
        data.patient_ids,
        n_splits=config.folds.n_splits,
        n_repeats=config.folds.n_repeats,
        seed=config.folds.seed,
    )
    save_folds(folds, data.patient_ids, config.folds.path)

    logger.info(
        "wrote %d folds over %d patients to %s",
        len(folds),
        data.n_patients,
        config.folds.path,
    )
    logger.info("class counts: %s", data.class_counts())
    return 0


def command_train(args: argparse.Namespace) -> int:
    """Train and score the transformer across the shared folds.

    Args:
        args: Parsed arguments carrying ``config``.

    Returns:
        Process exit status.
    """
    from mogformer.evaluation.runner import run_cross_validation

    config = _load(args.config)
    per_fold = run_cross_validation(config, models=args.models)
    logger.info("scored %d model-fold-metric rows", len(per_fold))
    return 0


def command_pretrain(args: argparse.Namespace) -> int:
    """Run outcome-blind self-supervised pretraining.

    Args:
        args: Parsed arguments carrying ``config``.

    Returns:
        Process exit status.
    """
    from mogformer.analysis.embedding import run_pretraining
    from mogformer.evaluation.runner import load_cohort

    config = _load(args.config)
    bundle = run_pretraining(config, load_cohort(config))
    logger.info(
        "pretraining finished: %s embeddings over %d genes, best monitoring loss %.4f",
        tuple(bundle.embeddings.shape),
        len(bundle.selected_genes),
        bundle.best_val,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser.

    Returns:
        A parser with one subcommand per phase.
    """
    parser = argparse.ArgumentParser(
        prog="mogformer",
        description=__doc__.splitlines()[0] if __doc__ else None,
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="emit debug-level logs"
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    for name, handler, help_text in (
        ("folds", command_folds, "build and persist the shared fold partition"),
        ("train", command_train, "train and score the supervised transformer"),
        ("pretrain", command_pretrain, "run outcome-blind self-supervised pretraining"),
    ):
        subcommand = subcommands.add_parser(name, help=help_text)
        subcommand.add_argument(
            "--config", required=True, help="path to the YAML configuration"
        )
        if name == "train":
            subcommand.add_argument(
                "--models",
                nargs="+",
                default=None,
                help="registry keys to run; omit to run every registered model",
            )
        subcommand.set_defaults(handler=handler)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``mogformer`` command.

    Args:
        argv: Argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit status.
    """
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)

    if not Path(args.config).exists():
        logger.error("configuration file not found: %s", args.config)
        return 1

    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
