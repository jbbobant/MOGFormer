"""Build the cohort bundles every downstream step reads.

A bundle is the aligned, cohort-level reduction of the cleaned inputs: patients
and genes aligned across omics, the interaction graph restricted to the gene
universe, and the clinical table. Nothing fitted to patients happens here;
gene selection, imputation, scaling and batch correction are fitted later,
inside each run, on its fitting patients only.

Inputs are the cleaned matrices under ``data/clean`` and ``data/METABRIC/clean``
(see ``docs/data.md``); outputs go to ``data/bundle/``.

Usage:
    uv run python -m scripts.build_cohort_bundles            # both cohorts
    uv run python -m scripts.build_cohort_bundles --cohort metabric
"""

from __future__ import annotations

import argparse
import sys

from utils.build_cohort_bundle import main as build_bundle

#: Bundles needed by the self-supervised pipeline, per cohort, in build order.
#: The TCGA self-supervised cohort is written into the TCGA bundle, which must
#: exist first (it holds the graph files the self-supervised cohort reuses).
BUNDLES: dict[str, tuple[str, ...]] = {
    "tcga": ("tcga", "tcga_ssl"),
    "metabric": ("metabric", "metabric_ssl"),
}


def main(argv: list[str] | None = None) -> int:
    """Build the bundles of the requested cohorts.

    Args:
        argv: Command-line arguments, or None for ``sys.argv``.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--cohort", choices=("tcga", "metabric", "both"), default="both"
    )
    args = parser.parse_args(argv)
    cohorts = list(BUNDLES) if args.cohort == "both" else [args.cohort]
    for cohort in cohorts:
        for bundle in BUNDLES[cohort]:
            status = build_bundle(["--cohort", bundle])
            if status != 0:
                return status
    return 0


if __name__ == "__main__":
    sys.exit(main())
