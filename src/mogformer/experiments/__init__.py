"""Experiments: the runs behind every reported result.

Each module here answers one question. They are ordinary library
code — linted, typed and tested like the rest of the package — rather than
scripts, so they cannot drift from the protocol the rest of the package
enforces.

Every function in this package writes its outputs to a directory and returns
what it wrote, so a run can be inspected afterwards without being repeated.
"""

from __future__ import annotations

from mogformer.experiments.benchmark import (
    BenchmarkResult,
    run_benchmark,
    run_benchmark_pair,
)
from mogformer.experiments.metabric import (
    ArchivedEncoder,
    load_archived_encoder,
    run_metabric_trans_sweep,
)
from mogformer.experiments.paths import (
    BundleLayout,
    provenance_record,
    rebase_config,
    save_json,
)
from mogformer.experiments.sign_test import rerun_sign_test
from mogformer.experiments.statistics import combine_per_fold, run_paired_statistics

__all__ = [
    "ArchivedEncoder",
    "BenchmarkResult",
    "BundleLayout",
    "combine_per_fold",
    "load_archived_encoder",
    "provenance_record",
    "rebase_config",
    "rerun_sign_test",
    "run_benchmark",
    "run_benchmark_pair",
    "run_metabric_trans_sweep",
    "run_paired_statistics",
    "save_json",
]
