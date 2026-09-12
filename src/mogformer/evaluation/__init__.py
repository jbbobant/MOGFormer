"""One harness: shared metrics, paired statistics and model registration."""

from __future__ import annotations

from mogformer.evaluation.metrics import (
    aggregate_across_folds,
    compute_fold_metrics,
    fold_confusion,
    nadeau_bengio_correction,
)
from mogformer.evaluation.plots import (
    OKABE_ITO,
    apply_house_style,
    plot_confusion,
    plot_critical_difference,
    plot_interval_forest,
    plot_model_comparison,
    plot_per_class_f1,
    save_figure,
)
from mogformer.evaluation.registry import (
    BalancedXGB,
    HarnessContext,
    LateFusionClassifier,
    ModelSpec,
    PAM50NearestCentroid,
    build_preprocessor,
    build_registry,
    late_fusion_spec,
)
from mogformer.evaluation.runner import (
    PRIMARY_METRIC,
    load_per_fold_scores,
    render_leaderboard,
    run_cross_validation,
    score_fold,
    summarise,
)
from mogformer.evaluation.stats import (
    corrected_resampled_ttest,
    friedman_nemenyi,
    pairwise_compare,
    rank_biserial,
)

__all__ = [
    "OKABE_ITO",
    "PRIMARY_METRIC",
    "BalancedXGB",
    "HarnessContext",
    "LateFusionClassifier",
    "ModelSpec",
    "PAM50NearestCentroid",
    "aggregate_across_folds",
    "apply_house_style",
    "build_preprocessor",
    "build_registry",
    "compute_fold_metrics",
    "corrected_resampled_ttest",
    "fold_confusion",
    "friedman_nemenyi",
    "late_fusion_spec",
    "load_per_fold_scores",
    "nadeau_bengio_correction",
    "pairwise_compare",
    "plot_confusion",
    "plot_critical_difference",
    "plot_interval_forest",
    "plot_model_comparison",
    "plot_per_class_f1",
    "rank_biserial",
    "render_leaderboard",
    "run_cross_validation",
    "save_figure",
    "score_fold",
    "summarise",
]
