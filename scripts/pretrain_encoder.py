"""Pretrain the self-supervised encoder and evaluate its representation.

Trains the default encoder of a cohort (``configs/ssl/<cohort>_default_encoder.yaml``)
for each seed, outcome-blind: no survival column is ever read. Each run fits its
gene axis, scalers, batch correction and gene programs on its fitting patients
(training plus monitoring); held-out patients are used only to report
reconstruction. The representation is then evaluated against PCA and MOFA+
by linear probes (fitted inside cross-validation folds), effective dimension,
seed stability and the clustering checkpoint.

Outputs:
    ``results/experiments/ssl_pretraining/<config name>_seed<seed>/`` per run;
    ``results/experiments/representation_evaluation/<config name>/``.

Usage:
    uv run python -m scripts.pretrain_encoder --cohort metabric
    uv run python -m scripts.pretrain_encoder --cohort tcga --seeds 42 --device cuda
    uv run python -m scripts.pretrain_encoder --cohort tcga --max-epochs 3  # smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from mogformer.cli.experiments import main as run_experiment

#: Bundle directory of each cohort's self-supervised cohort.
BUNDLES = {"tcga": "data/bundle/tcga", "metabric": "data/bundle/metabric/ssl"}

#: Default encoder configuration of each cohort.
CONFIGS = {
    "tcga": "configs/ssl/tcga_default_encoder.yaml",
    "metabric": "configs/ssl/metabric_default_encoder.yaml",
}


def main(argv: list[str] | None = None) -> int:
    """Pretrain one cohort's encoder over several seeds, then evaluate it.

    Args:
        argv: Command-line arguments, or None for ``sys.argv``.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cohort", choices=tuple(CONFIGS), required=True)
    parser.add_argument(
        "--config", help="encoder config; the cohort default if omitted"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--device", default="cuda", help="torch device")
    parser.add_argument(
        "--max-epochs", type=int, help="cap the epochs, for a smoke test only"
    )
    parser.add_argument(
        "--skip-evaluation", action="store_true", help="pretrain without evaluating"
    )
    args = parser.parse_args(argv)
    config = args.config or CONFIGS[args.cohort]
    bundle = BUNDLES[args.cohort]

    pretrain = [
        "ssl_pretraining",
        "--config",
        config,
        "--bundle",
        bundle,
        "--device",
        args.device,
        "--seeds",
        *map(str, args.seeds),
    ]
    if args.max_epochs:
        pretrain += ["--max-epochs", str(args.max_epochs)]
    status = run_experiment(pretrain)
    if status != 0 or args.skip_evaluation or args.max_epochs:
        return status

    name = yaml.safe_load(Path(config).read_text(encoding="utf-8"))["name"]
    runs = [f"results/experiments/ssl_pretraining/{name}_seed{s}" for s in args.seeds]
    return run_experiment(
        [
            "representation_evaluation",
            "--bundle",
            bundle,
            "--cohort",
            args.cohort,
            "--name",
            name,
            "--encoder-runs",
            *runs,
        ]
    )


if __name__ == "__main__":
    sys.exit(main())
