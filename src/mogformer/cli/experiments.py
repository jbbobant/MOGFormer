"""One entry point for every experiment behind the reported results.

Takes a cohort bundle, a results directory and the name of an experiment, and
does the rest — rebasing paths, running, and writing every table, figure and
provenance record under the results directory, so each run is one
self-describing folder.

Usage::

    python -m mogformer.cli.experiments list
    python -m mogformer.cli.experiments baselines --bundle DATA --out OUT
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from mogformer.config import ExperimentConfig, load_config, save_config
from mogformer.experiments.benchmark import (
    TRANSFORMER_MODELS,
    baseline_models_for,
    run_benchmark,
)
from mogformer.experiments.encoder_ablation import run_encoder_ablation
from mogformer.experiments.interpretability import run_interpretability
from mogformer.experiments.metabric import run_metabric_trans_sweep
from mogformer.experiments.outcome_analysis import run_outcome_analysis
from mogformer.experiments.paper_figures import build_paper_figures
from mogformer.experiments.paths import (
    BundleLayout,
    provenance_record,
    rebase_config,
    save_json,
)
from mogformer.experiments.patient_stratification import run_patient_stratification
from mogformer.experiments.report_figures import (
    build_report_figures,
    fold_encoding_separation,
)
from mogformer.experiments.representation_evaluation import (
    run_copy_number_scale_check,
    run_representation_evaluation,
)
from mogformer.experiments.sign_test import rerun_sign_test
from mogformer.experiments.simulation_benchmark import run_simulation_benchmark
from mogformer.experiments.ssl_pretraining import run_ssl_pretraining
from mogformer.experiments.statistics import (
    combine_per_fold,
    run_paired_statistics,
)
from mogformer.training.seed import seed_everything

logger = logging.getLogger("mogformer.experiments")

#: Bundle directory used when none is given, as written by
#: ``utils.build_cohort_bundle``.
DEFAULT_BUNDLE = Path("data/bundle/tcga")

#: Results directory used when none is given.
DEFAULT_OUTPUT = Path("results/experiments")


@dataclass(frozen=True)
class Experiment:
    """One runnable experiment.

    Attributes:
        key: Name typed on the command line.
        summary: What the run answers, in one line.
        needs_gpu: Whether this is worth spending GPU time on.
        run: The callable, taking the parsed arguments.
    """

    key: str
    summary: str
    needs_gpu: bool
    run: Callable[[argparse.Namespace], dict[str, object]]


def resolve_bundle(argument: str | None) -> BundleLayout:
    """Locate the cohort bundle.

    Args:
        argument: An explicit path, or None for :data:`DEFAULT_BUNDLE`.

    Returns:
        The layout. Which cohort archive a run reads depends on its
        configuration, so completeness is checked per run, in :func:`prepare`.

    Raises:
        FileNotFoundError: If there is no bundle at the location.
    """
    root = Path(argument) if argument else DEFAULT_BUNDLE
    if not (root / "MANIFEST.json").exists():
        raise FileNotFoundError(
            f"no bundle at {root}; build it with "
            "`uv run python -m utils.build_cohort_bundle` or pass --bundle"
        )
    return BundleLayout(root)


def resolve_output(argument: str | None) -> Path:
    """Choose where results are written.

    Args:
        argument: An explicit path, or None for :data:`DEFAULT_OUTPUT`.

    Returns:
        The directory, created if needed.
    """
    destination = Path(argument) if argument else DEFAULT_OUTPUT
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def prepare(
    config_path: str | Path,
    layout: BundleLayout,
    output: Path,
    args: argparse.Namespace,
    cohort_file: str | None = None,
) -> tuple[ExperimentConfig, BundleLayout]:
    """Load a configuration, point it at the bundle, apply any truncation.

    The truncation options exist so the whole path can be exercised in a minute
    before hours of GPU time are committed to it. They are recorded in the
    resolved configuration written beside the results, so a truncated run can
    never be mistaken for a full one.

    Args:
        config_path: Configuration file to load.
        layout: Where the bundle's files sit.
        output: Root the run writes under.
        args: Parsed command-line arguments, read for the truncation options.
        cohort_file: Override the bundle's cohort archive.

    Returns:
        The rebased configuration, also saved beside the results, and the
        layout naming the cohort archive it was rebased onto. The run must be
        given that layout: it is where the data are read from.
    """
    if cohort_file:
        layout = BundleLayout(layout.root, cohort_file=cohort_file)
    config = rebase_config(load_config(config_path), layout.require(), output)

    truncated = []
    if getattr(args, "n_repeats", None):
        config.folds.n_repeats_used = args.n_repeats
        truncated.append(f"{args.n_repeats}rep")
    if getattr(args, "max_folds", None):
        config.folds.max_folds = args.max_folds
        truncated.append(f"{args.max_folds}fold")
    if truncated:
        config.name = f"{config.name}__smoke_{'_'.join(truncated)}"
        logger.warning("TRUNCATED RUN: %s", ", ".join(truncated))
    if getattr(args, "max_epochs", None):
        config.train.max_epochs = args.max_epochs
        config.train.patience = min(config.train.patience, args.max_epochs)
        logger.warning("TRUNCATED RUN: %d epochs only", args.max_epochs)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, config.output_dir / "config.resolved.yaml")
    return config, layout


def experiment_baselines(args: argparse.Namespace) -> dict[str, object]:
    """Score every classical baseline on the shared partition.

    Every configuration given runs on the same partition, so the five-class,
    four-class and modality-subset leaderboards are directly comparable and
    their per-fold vectors pairable.

    Args:
        args: Parsed command-line arguments.

    Returns:
        A record naming the leaderboards written.
    """
    layout = resolve_bundle(args.bundle)
    output = resolve_output(args.out) / "baselines"
    written: dict[str, object] = {}

    for config_path in args.configs:
        cohort_file = args.cohort_file or _cohort_for(load_config(config_path))
        config, run_layout = prepare(config_path, layout, output, args, cohort_file)
        result = run_benchmark(
            config,
            run_layout,
            tuple(
                model
                for model in baseline_models_for(config)
                if not args.models or model in args.models
            ),
            output,
            device=args.device,
            n_search_jobs=args.search_jobs,
        )
        written[result.name] = {
            "output_dir": str(result.output_dir),
            "leaderboard": result.leaderboard().to_dict(orient="records"),
        }
    return written


def experiment_transformer_variants(args: argparse.Namespace) -> dict[str, object]:
    """Score the transformer variants at the full 25-estimate protocol.

    The archived ablation's best-looking runs stopped at five or ten estimates,
    which is one partition rather than a distribution over partitions. These
    re-runs take the top variants the whole way, on the same folds every
    baseline used.

    Args:
        args: Parsed command-line arguments.

    Returns:
        A record naming the leaderboards written.
    """
    layout = resolve_bundle(args.bundle)
    output = resolve_output(args.out) / "transformer_variants"
    written: dict[str, object] = {}

    for config_path in args.variants:
        config = load_config(config_path)
        cohort_file = args.cohort_file or _cohort_for(config)
        prepared, run_layout = prepare(config_path, layout, output, args, cohort_file)
        started = time.perf_counter()
        result = run_benchmark(
            prepared,
            run_layout,
            TRANSFORMER_MODELS,
            output,
            device=args.device,
            n_search_jobs=args.search_jobs,
        )
        written[result.name] = {
            "output_dir": str(result.output_dir),
            "cohort_file": cohort_file,
            "minutes": round((time.perf_counter() - started) / 60, 1),
            "leaderboard": result.leaderboard().to_dict(orient="records"),
        }
    return written


def experiment_paired_statistics(args: argparse.Namespace) -> dict[str, object]:
    """Compare every scored model pairwise on the shared partition.

    The benchmark runs are launched one variant at a time, so several per-fold
    files may be given at once. They are combined first, with any model key that
    appears under more than one run qualified by that run — three transformer
    variants all register as ``M_mogformer``, and a naive concatenation would
    fuse them into one impossible vector.

    Args:
        args: Parsed command-line arguments.

    Returns:
        The paired-test record.

    Raises:
        ValueError: If no per-fold file could be found.
    """
    output = resolve_output(args.out) / "paired_statistics" / args.comparison
    sources = [Path(pattern) for pattern in args.per_fold]
    if args.search:
        sources += sorted(
            path
            for path in resolve_output(args.out).rglob("metrics_per_fold.csv")
            if "smoke" not in str(path)
        )
    resolved = sorted({path for path in sources if path.exists()})
    if not resolved:
        raise ValueError(
            "no metrics_per_fold.csv found; pass --per-fold or run a benchmark first"
        )
    logger.info("combining %d per-fold file(s)", len(resolved))

    combine_per_fold(
        resolved,
        output / "metrics_per_fold_combined.csv",
        exclude_models=args.exclude_models,
    )
    return run_paired_statistics(
        output / "metrics_per_fold_combined.csv",
        output,
        metric=args.metric,
        n_splits=args.n_splits,
    )


def experiment_sign_concordance(args: argparse.Namespace) -> dict[str, object]:
    """Re-test trans sign concordance under the corrected null.

    Args:
        args: Parsed command-line arguments.

    Returns:
        A record per cohort tested.
    """
    output = resolve_output(args.out) / "sign_concordance"
    written: dict[str, object] = {}
    for entry in args.edges:
        cohort, _, path = entry.partition("=")
        if not path:
            cohort, path = "cohort", entry
        written[cohort] = rerun_sign_test(
            path,
            output / cohort,
            cohort=cohort,
            n_permutations=args.n_permutations,
        )
    return written


def experiment_external_trans_sweep(args: argparse.Namespace) -> dict[str, object]:
    """Measure the external cohort's trans effects and test their signs.

    The TCGA half of the sign test is a re-analysis of stored slopes. This half
    is not: METABRIC's archived run saved co-expression rather than slopes, so
    the sweep has to be measured again from the frozen encoder.

    Args:
        args: Parsed command-line arguments.

    Returns:
        The sweep record, including the corrected sign test.
    """
    run_dir = args.run_dir
    if not run_dir:
        run_dir = resolve_bundle(args.bundle).encoder(args.cohort)
    output = resolve_output(args.out) / "external_trans_sweep" / args.cohort
    return run_metabric_trans_sweep(
        run_dir,
        output,
        device=args.device,
        with_permuted_null=not args.no_permuted_null,
        max_factors=args.max_factors,
    )


def experiment_simulation_benchmark(args: argparse.Namespace) -> dict[str, object]:
    """Score sign recovery on cohorts simulated from the curated regulatory graph.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Where the summary table was written.
    """
    output = resolve_output(args.out) / "simulation_benchmark"
    table = run_simulation_benchmark(
        args.plan,
        resolve_bundle(args.bundle).encoder(args.cohort),
        output,
        device=args.device,
        only=args.conditions,
        max_epochs=args.max_epochs,
        max_factors=args.max_factors,
    )
    return {"summary": str(output / "summary.csv"), "rows": len(table)}


def experiment_report_figures(args: argparse.Namespace) -> dict[str, object]:
    """Draw the benchmark figures from the saved per-fold scores.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Mapping of figure name to the written PNG.
    """
    output = resolve_output(args.out) / "report_figures"
    separation = None
    if not args.no_encodings:
        raw = load_config(args.encoding_config)
        layout = BundleLayout(
            resolve_bundle(args.bundle).root, cohort_file=_cohort_for(raw)
        )
        config = rebase_config(raw, layout, output / "encoding_run")
        separation = fold_encoding_separation(config, layout)
    written: dict[str, object] = dict(
        build_report_figures(args.results_root, output, nearest_distance=separation)
    )
    paper = resolve_output(args.out) / "paper_figures"
    written.update(build_paper_figures(args.results_root, paper))
    return written


def _cohort_for(config: ExperimentConfig) -> str:
    """Choose the bundle archive a configuration's methylation source implies.

    Args:
        config: Configuration as written in ``configs/``.

    Returns:
        The archive filename to read.
    """
    return "cohort_m.npz" if "_M" in config.data.methy_file else "cohort_beta.npz"


def experiment_representation_evaluation(args: argparse.Namespace) -> dict[str, object]:
    """Score frozen encoders' read-outs against equal-size comparators.

    Args:
        args: Parsed command-line arguments.

    Returns:
        The evaluation record.
    """
    return run_representation_evaluation(
        args.encoder_runs,
        resolve_bundle(args.bundle).root,
        resolve_output(args.out) / "representation_evaluation",
        cohort=args.cohort,
        name=args.name,
        n_resample=args.n_resample,
        with_mofa=not args.no_mofa,
    )


def experiment_ssl_pretraining(args: argparse.Namespace) -> dict[str, object]:
    """Pretrain one self-supervised configuration once per seed.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Per run: directory, epochs, timing and participation ratios.
    """
    return run_ssl_pretraining(
        args.config,
        resolve_bundle(args.bundle).root,
        resolve_output(args.out) / "ssl_pretraining",
        seeds=args.seeds,
        device=args.device,
        max_epochs=args.max_epochs,
    )


def experiment_encoder_ablation(args: argparse.Namespace) -> dict[str, object]:
    """Judge each tested group of encoder changes by leaving it out.

    Args:
        args: Parsed command-line arguments.

    Returns:
        The verdict per group.
    """
    root = resolve_output(args.out)
    return run_encoder_ablation(
        root / "representation_evaluation",
        root / "encoder_ablation",
        cohort=args.cohort,
        reference=args.reference,
        prefix=args.prefix,
    )


def experiment_patient_stratification(args: argparse.Namespace) -> dict[str, object]:
    """Carry out the pre-registered stratification protocol, structure only.

    Args:
        args: Parsed command-line arguments.

    Returns:
        The per-population record.
    """
    root = resolve_output(args.out)
    return run_patient_stratification(
        args.protocol,
        root / "ssl_pretraining",
        root / args.name,
    )


def experiment_outcome_analysis(args: argparse.Namespace) -> dict[str, object]:
    """Carry out the locked outcome protocol (or its permuted-outcome check).

    Args:
        args: Parsed command-line arguments.

    Returns:
        The analysis record.
    """
    root = resolve_output(args.out)
    name = "outcome_analysis_permuted" if args.permuted_outcomes else "outcome_analysis"
    if args.post_hoc:
        name += "_post_hoc"
    return run_outcome_analysis(
        args.protocol,
        root / "ssl_pretraining",
        root / args.stratification,
        root / args.stratification / name,
        permute=args.permuted_outcomes,
        post_hoc=args.post_hoc,
    )


def experiment_interpretability(args: argparse.Namespace) -> dict[str, object]:
    """Attribute the encoder's risk and stratum scores, and read its programs.

    Args:
        args: Parsed command-line arguments.

    Returns:
        The interpretability record.
    """
    root = resolve_output(args.out)
    return run_interpretability(
        args.protocol,
        root / "ssl_pretraining",
        root / args.stratification,
        root / args.stratification / "interpretability",
    )


def experiment_copy_number_scale_check(args: argparse.Namespace) -> dict[str, object]:
    """Score the comparators on absolute and on ploidy-relative copy number.

    Args:
        args: Parsed command-line arguments.

    Returns:
        The check's record.
    """
    return run_copy_number_scale_check(
        resolve_bundle(args.bundle).root,
        resolve_output(args.out) / "representation_evaluation",
        n_resample=args.n_resample,
    )


#: Every runnable experiment, upstream results first.
EXPERIMENTS: tuple[Experiment, ...] = (
    Experiment(
        key="baselines",
        summary="classical baselines on the shared partition, both class configs",
        needs_gpu=False,
        run=experiment_baselines,
    ),
    Experiment(
        key="transformer_variants",
        summary="transformer variants at the full 25-estimate protocol",
        needs_gpu=True,
        run=experiment_transformer_variants,
    ),
    Experiment(
        key="paired_statistics",
        summary="paired Wilcoxon and Friedman-Nemenyi over per-fold scores",
        needs_gpu=False,
        run=experiment_paired_statistics,
    ),
    Experiment(
        key="external_trans_sweep",
        summary="re-measure a cohort's trans sweep from its frozen encoder",
        needs_gpu=True,
        run=experiment_external_trans_sweep,
    ),
    Experiment(
        key="sign_concordance",
        summary="trans sign concordance against the corrected null",
        needs_gpu=False,
        run=experiment_sign_concordance,
    ),
    Experiment(
        key="simulation_benchmark",
        summary="sign recovery on cohorts simulated from the curated graph",
        needs_gpu=True,
        run=experiment_simulation_benchmark,
    ),
    Experiment(
        key="ssl_pretraining",
        summary="outcome-blind masked pretraining of the encoder, one run per seed",
        needs_gpu=True,
        run=experiment_ssl_pretraining,
    ),
    Experiment(
        key="encoder_ablation",
        summary="leave-one-group-out verdicts from the suite's evaluations",
        needs_gpu=False,
        run=experiment_encoder_ablation,
    ),
    Experiment(
        key="representation_evaluation",
        summary="probes, effective dimension and clustering checkpoint of an encoder",
        needs_gpu=False,
        run=experiment_representation_evaluation,
    ),
    Experiment(
        key="patient_stratification",
        summary="pre-registered consensus stratification, ladder and audit",
        needs_gpu=False,
        run=experiment_patient_stratification,
    ),
    Experiment(
        key="outcome_analysis",
        summary="locked survival analysis of the strata and the embedding",
        needs_gpu=False,
        run=experiment_outcome_analysis,
    ),
    Experiment(
        key="interpretability",
        summary="attribution of risk and strata to genes, omics reliance, programs",
        needs_gpu=True,
        run=experiment_interpretability,
    ),
    Experiment(
        key="copy_number_scale_check",
        summary="comparators on absolute vs ploidy-relative copy number",
        needs_gpu=False,
        run=experiment_copy_number_scale_check,
    ),
    Experiment(
        key="report_figures",
        summary="benchmark, modality and gene-identity figures from saved scores",
        needs_gpu=False,
        run=experiment_report_figures,
    ),
)


def build_experiment_parser() -> argparse.ArgumentParser:
    """Construct the command-line interface.

    Returns:
        The parser, with one subcommand per experiment plus ``list``.
    """
    parser = argparse.ArgumentParser(
        prog="mogformer-experiments",
        description="Run one experiment and save everything it produced.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    subcommands = parser.add_subparsers(dest="command", required=True)

    subcommands.add_parser("list", help="show the runnable experiments")

    for experiment in EXPERIMENTS:
        marker = " [GPU]" if experiment.needs_gpu else ""
        sub = subcommands.add_parser(
            experiment.key, help=f"{experiment.summary}{marker}"
        )
        sub.add_argument("--bundle", help="cohort bundle directory")
        sub.add_argument("--out", help="directory to write results under")
        sub.add_argument("--device", help="torch device, e.g. cuda or cpu")
        sub.add_argument(
            "--search-jobs",
            dest="search_jobs",
            type=int,
            default=-1,
            help="processes per inner hyperparameter search; -1 uses every core",
        )
        sub.add_argument(
            "--cohort-file",
            dest="cohort_file",
            help="cohort archive inside the bundle, overriding the default",
        )
        sub.add_argument(
            "--n-repeats",
            dest="n_repeats",
            type=int,
            help="use only the first N repeats; for smoke tests, never results",
        )
        sub.add_argument(
            "--max-folds",
            dest="max_folds",
            type=int,
            help="use only the first N folds of each repeat; smoke tests only",
        )
        sub.add_argument(
            "--max-epochs",
            dest="max_epochs",
            type=int,
            help="cap training epochs; for smoke tests, never results",
        )

        if experiment.key == "baselines":
            sub.add_argument(
                "--configs",
                nargs="+",
                default=["configs/tcga.yaml", "configs/tcga_4class.yaml"],
                help="one configuration file per cohort or modality subset",
            )
            sub.add_argument(
                "--models",
                nargs="+",
                help="score only these registry keys; all applicable ones when omitted",
            )
        if experiment.key == "transformer_variants":
            sub.add_argument(
                "--variants",
                nargs="+",
                required=True,
                help="one configuration file per variant to run",
            )
        if experiment.key == "report_figures":
            sub.add_argument(
                "--results-root",
                dest="results_root",
                default="results/experiments",
                help="directory holding baselines/ and transformer_variants/",
            )
            sub.add_argument(
                "--encoding-config",
                dest="encoding_config",
                default="configs/transformer_variants/attention_no_gene_identity_5class.yaml",
                help="configuration whose fold-0 encodings the identity figure shows",
            )
            sub.add_argument(
                "--no-encodings",
                dest="no_encodings",
                action="store_true",
                help="skip the positional-encoding panel, which reads the cohort",
            )
        if experiment.key == "paired_statistics":
            sub.add_argument(
                "--per-fold",
                dest="per_fold",
                nargs="*",
                default=[],
                help="metrics_per_fold.csv files written by benchmark runs",
            )
            sub.add_argument(
                "--search",
                action="store_true",
                help="also use every non-smoke per-fold file under --out",
            )
            sub.add_argument(
                "--comparison",
                default="all",
                help="name of this comparison, used as its output directory",
            )
            sub.add_argument(
                "--exclude-models",
                dest="exclude_models",
                nargs="*",
                default=["B0a_dummy_stratified", "B0b_dummy_mostfreq"],
                help="registry keys to leave out of the comparison",
            )
            sub.add_argument("--metric", default="macro_f1")
            sub.add_argument("--n-splits", dest="n_splits", type=int, default=5)
        if experiment.key == "external_trans_sweep":
            sub.add_argument(
                "--cohort",
                default="metabric",
                help="which archived encoder in the bundle to probe",
            )
            sub.add_argument(
                "--run-dir",
                dest="run_dir",
                help="an archived run outside the bundle, overriding --cohort",
            )
            sub.add_argument(
                "--max-factors",
                dest="max_factors",
                type=int,
                help="probe only the first N factors; smoke tests only",
            )
            sub.add_argument(
                "--no-permuted-null",
                dest="no_permuted_null",
                action="store_true",
                help="skip the permuted-identity control, halving the runtime",
            )
        if experiment.key == "simulation_benchmark":
            sub.add_argument(
                "--plan",
                type=Path,
                default=Path("configs/simulation_benchmark.yaml"),
                help="the sweep: simulation settings, probe settings, conditions",
            )
            sub.add_argument(
                "--cohort",
                default="tcga",
                help="archived encoder whose settings and gene axis are reused",
            )
            sub.add_argument(
                "--conditions",
                nargs="+",
                help="run only these conditions, by directory name",
            )
            sub.add_argument(
                "--max-factors",
                dest="max_factors",
                type=int,
                help="probe only the first N factors; smoke tests only",
            )
        if experiment.key == "interpretability":
            sub.add_argument(
                "--protocol",
                default="configs/patient_stratification/protocol_combat.yaml",
                help="the stratification protocol naming the encoder runs",
            )
            sub.add_argument(
                "--stratification",
                default="patient_stratification_combat",
                help="output directory of the stratification whose strata are used",
            )
        if experiment.key == "outcome_analysis":
            sub.add_argument(
                "--protocol",
                default="configs/patient_stratification/protocol.yaml",
                help="the stratification protocol's settings",
            )
            sub.add_argument(
                "--stratification",
                default="patient_stratification",
                help="output directory of the stratification whose strata are analysed",
            )
            sub.add_argument(
                "--post-hoc",
                action="store_true",
                help="analyse strata that missed the bar; outputs labelled post hoc",
            )
            sub.add_argument(
                "--permuted-outcomes",
                action="store_true",
                help="run on permuted outcomes: the pipeline check before the real run",
            )
        if experiment.key == "patient_stratification":
            sub.add_argument(
                "--protocol",
                default="configs/patient_stratification/protocol.yaml",
                help="the protocol's settings",
            )
            sub.add_argument(
                "--name",
                default="patient_stratification",
                help="output directory under the results root",
            )
        if experiment.key == "encoder_ablation":
            sub.add_argument("--cohort", choices=("tcga", "metabric"), required=True)
            sub.add_argument(
                "--reference",
                required=True,
                help="evaluation name of the reference encoder, e.g. tcga_ssl_encoder",
            )
            sub.add_argument(
                "--prefix",
                required=True,
                help="evaluation-name prefix of the ablation runs, e.g. tcga_encoder",
            )
        if experiment.key == "ssl_pretraining":
            sub.add_argument(
                "--config",
                type=Path,
                required=True,
                help="self-supervised configuration, e.g. configs/ssl/tcga.yaml",
            )
            sub.add_argument(
                "--seeds",
                type=int,
                nargs="+",
                default=[42],
                help="training seeds, one run each; the splits do not change",
            )
        if experiment.key == "representation_evaluation":
            sub.add_argument(
                "--encoder-runs",
                dest="encoder_runs",
                type=Path,
                nargs="+",
                default=[Path("results/SSL/SURV_CLEAN/TCGA/E600_p60_gated_meta")],
                help="pretraining runs of one configuration, one per seed",
            )
            sub.add_argument(
                "--cohort",
                choices=("tcga", "metabric"),
                default="tcga",
                help="cohort whose targets and populations are used",
            )
            sub.add_argument(
                "--name", help="output subdirectory; the first run's name if omitted"
            )
            sub.add_argument(
                "--no-mofa",
                dest="no_mofa",
                action="store_true",
                help="leave MOFA+ out of the comparators (it takes minutes)",
            )
        if experiment.key in ("representation_evaluation", "copy_number_scale_check"):
            sub.add_argument(
                "--n-resample",
                dest="n_resample",
                type=int,
                default=200,
                help="consensus resamples per K in the clustering checkpoint",
            )
        if experiment.key == "sign_concordance":
            sub.add_argument(
                "--edges",
                nargs="+",
                required=True,
                help="per-edge slope tables, as NAME=PATH or PATH",
            )
            sub.add_argument(
                "--n-permutations",
                dest="n_permutations",
                type=int,
                default=10_000,
            )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one experiment.

    Args:
        argv: Command-line arguments, or None for ``sys.argv``.

    Returns:
        Process exit status.
    """
    args = build_experiment_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )

    if args.command == "list":
        for experiment in EXPERIMENTS:
            marker = "GPU" if experiment.needs_gpu else "CPU"
            print(f"{experiment.key:18} [{marker}] {experiment.summary}")
        return 0

    experiment = next(item for item in EXPERIMENTS if item.key == args.command)
    seed_everything(42)
    started = time.perf_counter()

    try:
        written = experiment.run(args)
    except (FileNotFoundError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    output = resolve_output(args.out) / experiment.key
    save_json(
        provenance_record(
            {
                "experiment": experiment.key,
                "minutes": round((time.perf_counter() - started) / 60, 1),
                "written": written,
            }
        ),
        output / "run_record.json",
    )
    logger.info(
        "%s finished in %.1f min; results under %s",
        experiment.key,
        (time.perf_counter() - started) / 60,
        output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
