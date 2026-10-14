"""Interpret the frozen encoder on the discovery population.

Attributes the relapse risk score (post hoc) and the high-risk stratum to every
input (gene x omics) with integrated gradients, reports the fusion gate's
reliance on each omics, and describes each gene program: the genes it attends
to and the relapse information its block carries alone.

The stratification named by ``--stratification`` must exist (see
``scripts/stratify_and_analyse_survival.py``).

Outputs: ``results/experiments/<stratification>/interpretability/``.

Usage:
    uv run python -m scripts.interpret_encoder
"""

from __future__ import annotations

import argparse
import sys

from mogformer.cli.experiments import main as run_experiment


def main(argv: list[str] | None = None) -> int:
    """Run the interpretability analyses.

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
        "--stratification",
        default="patient_stratification_combat",
        help="output directory of the stratification whose strata are explained",
    )
    args = parser.parse_args(argv)
    return run_experiment(
        [
            "interpretability",
            "--protocol",
            args.protocol,
            "--stratification",
            args.stratification,
        ]
    )


if __name__ == "__main__":
    sys.exit(main())
