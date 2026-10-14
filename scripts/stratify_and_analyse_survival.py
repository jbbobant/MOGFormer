"""Stratify patients, then analyse survival under the locked protocols.

1. **Stratification** (structure only; no outcome column is read): consensus
   clustering of each population's embedding at K = 2-6 with k-means and a
   Gaussian mixture, the pre-registered stability bar, the representation
   ladder, the audit against known axes, and the replication of the discovery
   strata in TCGA. Protocol: ``configs/patient_stratification/README.md``.
2. **Pipeline check on permuted outcomes**: the full survival analysis with
   relapse and death shuffled across patients; every association must vanish.
3. **Survival analysis** of METABRIC ER+/HER2-, as locked in
   ``configs/patient_stratification/outcome_protocol.md``: strata (when they
   meet the bar, or with ``--post-hoc``) and the continuous embedding against
   PCA, MOFA+ and the clinical model.

The pretraining runs named in the protocol file must exist (see
``scripts/pretrain_encoder.py``).

Usage:
    uv run python -m scripts.stratify_and_analyse_survival
    uv run python -m scripts.stratify_and_analyse_survival --post-hoc
    uv run python -m scripts.stratify_and_analyse_survival --name DIR --protocol YAML
"""

from __future__ import annotations

import argparse
import sys

from mogformer.cli.experiments import main as run_experiment


def main(argv: list[str] | None = None) -> int:
    """Run the stratification, the permuted check and the survival analysis.

    Args:
        argv: Command-line arguments, or None for ``sys.argv``.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--protocol",
        default="configs/patient_stratification/protocol_combat.yaml",
        help="stratification settings naming the encoder runs",
    )
    parser.add_argument(
        "--name",
        default="patient_stratification_combat",
        help="output directory under results/experiments",
    )
    parser.add_argument(
        "--post-hoc",
        action="store_true",
        help="analyse strata that missed the bar (outputs labelled post hoc)",
    )
    parser.add_argument(
        "--skip-stratification",
        action="store_true",
        help="reuse an existing stratification",
    )
    args = parser.parse_args(argv)

    if not args.skip_stratification:
        status = run_experiment(
            ["patient_stratification", "--protocol", args.protocol, "--name", args.name]
        )
        if status != 0:
            return status
    outcome = [
        "outcome_analysis",
        "--protocol",
        args.protocol,
        "--stratification",
        args.name,
    ]
    if args.post_hoc:
        outcome.append("--post-hoc")
    status = run_experiment([*outcome, "--permuted-outcomes"])
    if status != 0:
        return status
    return run_experiment(outcome)


if __name__ == "__main__":
    sys.exit(main())
