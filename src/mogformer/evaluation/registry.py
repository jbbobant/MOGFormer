"""Registry of every model the harness can score.

One registry, one set of folds, one metrics module: that is what makes the
comparison between a gradient-boosted tree and a graph transformer a genuine
paired test rather than two runs that happen to report the same number.

Every entry returns a complete scikit-learn pipeline carrying its **own**
preprocessing front-end. Preprocessing therefore refits inside each fold, and
inside each inner hyperparameter-search fold too, which is what keeps the
protocol leakage-free no matter which model is being scored.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

from mogformer.data.omics import MODALITY_ORDER
from mogformer.data.preprocess import MultiOmicsTransformer

logger = logging.getLogger(__name__)


@dataclass
class HarnessContext:
    """Everything a model factory needs to build its pipeline for one run.

    Attributes:
        n_genes: Genes per modality block in the raw matrix.
        gene_names: Gene symbols in block column order.
        curated_genes: Genes force-included after variance ranking.
        pam50_genes: The label-defining gene panel, used by the centroid
            reference model only.
        n_classes: Number of subtypes.
        top_k: Size of the variance-ranked pool.
        active_modalities: Modalities in play for this run.
        svm_kernel: Kernel for the support-vector baseline.
        xgb_device: Device string for the boosted-tree baseline.
        seed: Seed passed to every stochastic estimator.
        mad_on_log_rna: Rank expression by the deviation of its log.
    """

    n_genes: int
    gene_names: Sequence[str]
    curated_genes: Sequence[str]
    pam50_genes: Sequence[str]
    n_classes: int
    top_k: int = 250
    active_modalities: Sequence[str] = MODALITY_ORDER
    svm_kernel: str = "linear"
    xgb_device: str = "cpu"
    seed: int = 42
    mad_on_log_rna: bool = False


def build_preprocessor(
    context: HarnessContext,
    modalities: Sequence[str],
    top_k: int | None = None,
    curated: Sequence[str] | None = None,
) -> MultiOmicsTransformer:
    """Construct a fold-local preprocessing stage for one model.

    Args:
        context: Run-level settings.
        modalities: Modalities this pipeline consumes.
        top_k: Override the pool size, for models that select differently.
        curated: Override the force-included genes.

    Returns:
        An unfitted transformer configured for this model.
    """
    return MultiOmicsTransformer(
        n_genes=context.n_genes,
        gene_names=context.gene_names,
        top_k=context.top_k if top_k is None else top_k,
        curated_genes=context.curated_genes if curated is None else curated,
        active_modalities=tuple(modalities),
        mad_on_log_rna=context.mad_on_log_rna,
    )


class BalancedXGB:
    """Gradient-boosted trees that rebalance classes at fit time.

    Deriving the sample weights inside ``fit`` rather than passing them in is
    what keeps the model correct under a randomised search, which subsets ``y``
    internally: weights computed once outside would describe the wrong subset.

    This is a thin subclass created lazily so that importing the registry does
    not require xgboost to be installed.
    """

    def __new__(cls, **kwargs: Any) -> Any:
        """Return a configured, class-balancing ``XGBClassifier``."""
        from xgboost import XGBClassifier

        class _BalancedXGB(XGBClassifier):
            """Recompute balanced weights from whatever subset it is given."""

            def fit(self, X: Any, y: Any, **fit_params: Any) -> Any:
                """Fit with class weights derived from ``y`` at call time."""
                weights = compute_sample_weight(class_weight="balanced", y=y)
                return super().fit(X, y, sample_weight=weights, **fit_params)

        return _BalancedXGB(**kwargs)


class PAM50NearestCentroid(BaseEstimator, ClassifierMixin):
    """Classify by rank correlation to per-class expression centroids.

    This is the conceptually load-bearing reference model, not merely another
    baseline. PAM50 labels are themselves produced by a nearest-centroid
    classifier over expression, so this reproduces the label-generating
    procedure. Any model that fails to beat it is not learning biology beyond
    the definition of the target.

    Attributes:
        classes_: Class labels seen during ``fit``.
        centroids_: Per-class mean expression profiles.
    """

    def __init__(self, temperature: float = 10.0) -> None:
        """Store the softmax temperature used to turn correlations into scores.

        Args:
            temperature: Higher values sharpen the probability distribution.
        """
        self.temperature = temperature

    @staticmethod
    def _rank_standardise(matrix: np.ndarray) -> np.ndarray:
        """Rank each row and scale it to unit norm.

        Working on ranks makes the correlation Spearman rather than Pearson,
        which is what the published panel uses and which is robust to the
        skewed dynamic range of expression data.

        Args:
            matrix: Values of shape ``(n_rows, n_features)``.

        Returns:
            Centred, unit-norm ranks of the same shape.
        """
        ranks = np.apply_along_axis(rankdata, 1, matrix).astype(float)
        ranks -= ranks.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(ranks, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return ranks / norms

    def fit(self, X: np.ndarray, y: np.ndarray) -> PAM50NearestCentroid:
        """Compute one centroid per class from the training fold.

        Args:
            X: Training features, shape ``(n_samples, n_features)``.
            y: Integer labels, shape ``(n_samples,)``.

        Returns:
            The fitted classifier.
        """
        self.classes_ = np.unique(y)
        self.centroids_ = np.vstack(
            [X[y == label].mean(axis=0) for label in self.classes_]
        )
        self._centroid_ranks = self._rank_standardise(self.centroids_)
        return self

    def _correlate(self, X: np.ndarray) -> np.ndarray:
        """Return each sample's rank correlation to every centroid."""
        return self._rank_standardise(np.asarray(X, dtype=float)) @ (
            self._centroid_ranks.T
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign each sample to its best-correlating centroid.

        Args:
            X: Features, shape ``(n_samples, n_features)``.

        Returns:
            Predicted labels, shape ``(n_samples,)``.
        """
        return self.classes_[np.argmax(self._correlate(X), axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return softmax-scaled correlations as class probabilities.

        Args:
            X: Features, shape ``(n_samples, n_features)``.

        Returns:
            Probabilities of shape ``(n_samples, n_classes)``.
        """
        scores = self._correlate(X) * self.temperature
        scores -= scores.max(axis=1, keepdims=True)
        exponentiated = np.exp(scores)
        return exponentiated / exponentiated.sum(axis=1, keepdims=True)


class LateFusionClassifier(BaseEstimator, ClassifierMixin):
    """Average the predictions of one independent model per modality.

    The counterpart to early fusion, where modalities are concatenated before
    the model sees them. Comparing the two says whether a model benefits from
    seeing modalities jointly or merely from seeing them at all.

    Attributes:
        classes_: Class labels seen during ``fit``.
        models_: The fitted per-modality pipelines.
    """

    def __init__(
        self,
        base_pipeline_factory: Callable[[str], Pipeline],
        modalities: Sequence[str],
    ) -> None:
        """Store the factory and the modalities to fuse.

        Args:
            base_pipeline_factory: Returns a fresh pipeline restricted to the
                named modality.
            modalities: Modalities to train separate models on.
        """
        self.base_pipeline_factory = base_pipeline_factory
        self.modalities = list(modalities)

    def fit(self, X: np.ndarray, y: np.ndarray) -> LateFusionClassifier:
        """Fit one pipeline per modality.

        Args:
            X: Training features carrying every modality block.
            y: Integer labels, shape ``(n_samples,)``.

        Returns:
            The fitted classifier.
        """
        self.classes_ = np.unique(y)
        self.models_ = {}
        for modality in self.modalities:
            pipeline = self.base_pipeline_factory(modality)
            pipeline.fit(X, y)
            self.models_[modality] = pipeline
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Average the per-modality probability estimates.

        Args:
            X: Features carrying every modality block.

        Returns:
            Probabilities of shape ``(n_samples, n_classes)``.
        """
        return np.mean(
            [self.models_[m].predict_proba(X) for m in self.modalities], axis=0
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the highest-probability class after fusion.

        Args:
            X: Features carrying every modality block.

        Returns:
            Predicted labels, shape ``(n_samples,)``.
        """
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


@dataclass
class ModelSpec:
    """One registered model and how to build, tune and describe it.

    Attributes:
        name: Registry key, also the label used in results tables.
        family: Coarse grouping, for plots.
        build: Factory taking a context and returning an unfitted pipeline.
        search_space: Parameter grid for the inner search; empty means no
            tuning.
        tune: Whether to run the inner search at all.
        notes: Why this model is in the suite.
    """

    name: str
    family: str
    build: Callable[[HarnessContext], BaseEstimator]
    search_space: dict[str, list[Any]] = field(default_factory=dict)
    tune: bool = False
    notes: str = ""


def build_registry() -> dict[str, ModelSpec]:
    """Construct the full model registry.

    Returns:
        Mapping of registry key to specification, ordered from the performance
        floor upward.
    """
    registry: dict[str, ModelSpec] = {}

    registry["B0a_dummy_stratified"] = ModelSpec(
        name="B0a_dummy_stratified",
        family="dummy",
        build=lambda c: Pipeline(
            [
                ("prep", build_preprocessor(c, c.active_modalities)),
                (
                    "clf",
                    DummyClassifier(strategy="stratified", random_state=c.seed),
                ),
            ]
        ),
        notes="performance floor, matching the class prior",
    )

    registry["B0b_dummy_mostfreq"] = ModelSpec(
        name="B0b_dummy_mostfreq",
        family="dummy",
        build=lambda c: Pipeline(
            [
                ("prep", build_preprocessor(c, c.active_modalities)),
                ("clf", DummyClassifier(strategy="most_frequent")),
            ]
        ),
        notes="performance floor, always the majority class",
    )

    registry["B1_pam50_centroid"] = ModelSpec(
        name="B1_pam50_centroid",
        family="centroid",
        build=lambda c: Pipeline(
            [
                # Expression only, panel genes only, no variance ranking.
                (
                    "prep",
                    build_preprocessor(c, ("rna",), top_k=1, curated=c.pam50_genes),
                ),
                ("clf", PAM50NearestCentroid()),
            ]
        ),
        notes="label-source reference: the target is itself RNA-derived",
    )

    registry["B2_elasticnet_logreg"] = ModelSpec(
        name="B2_elasticnet_logreg",
        family="linear",
        build=lambda c: Pipeline(
            [
                ("prep", build_preprocessor(c, c.active_modalities)),
                (
                    "clf",
                    LogisticRegression(
                        penalty="elasticnet",
                        solver="saga",
                        max_iter=4000,
                        class_weight="balanced",
                        random_state=c.seed,
                    ),
                ),
            ]
        ),
        search_space={
            "clf__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
            "clf__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        },
        tune=True,
    )

    registry["B3_linear_svm"] = ModelSpec(
        name="B3_linear_svm",
        family="svm",
        build=lambda c: Pipeline(
            [
                ("prep", build_preprocessor(c, c.active_modalities)),
                (
                    "clf",
                    SVC(
                        kernel=c.svm_kernel,
                        probability=True,
                        class_weight="balanced",
                        random_state=c.seed,
                        decision_function_shape="ovr",
                    ),
                ),
            ]
        ),
        search_space={"clf__C": [0.01, 0.1, 1.0, 10.0]},
        tune=True,
    )

    registry["B4_random_forest"] = ModelSpec(
        name="B4_random_forest",
        family="rf",
        build=lambda c: Pipeline(
            [
                ("prep", build_preprocessor(c, c.active_modalities)),
                (
                    "clf",
                    RandomForestClassifier(
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=c.seed,
                    ),
                ),
            ]
        ),
        search_space={
            "clf__n_estimators": [300, 600],
            "clf__max_depth": [None, 8, 16],
            "clf__max_features": ["sqrt", 0.1],
        },
        tune=True,
    )

    registry["B5_xgboost"] = ModelSpec(
        name="B5_xgboost",
        family="gbt",
        build=lambda c: Pipeline(
            [
                ("prep", build_preprocessor(c, c.active_modalities)),
                (
                    "clf",
                    BalancedXGB(
                        objective="multi:softprob",
                        num_class=c.n_classes,
                        eval_metric="mlogloss",
                        tree_method="hist",
                        device=c.xgb_device,
                        random_state=c.seed,
                    ),
                ),
            ]
        ),
        search_space={
            "clf__max_depth": [3, 4, 6],
            "clf__n_estimators": [300, 600],
            "clf__learning_rate": [0.03, 0.1],
            "clf__subsample": [0.7, 1.0],
            "clf__colsample_bytree": [0.5, 0.8],
        },
        tune=True,
        notes="strongest classical model in this regime",
    )

    return registry


def late_fusion_spec(name: str, base: ModelSpec) -> ModelSpec:
    """Wrap a registered model as its late-fusion counterpart.

    Args:
        name: Registry key of the base model.
        base: The specification to wrap.

    Returns:
        A specification training one copy of the base model per modality and
        averaging their probabilities.
    """

    def build(context: HarnessContext) -> BaseEstimator:
        def factory(modality: str) -> Pipeline:
            single = HarnessContext(
                **{**context.__dict__, "active_modalities": (modality,)}
            )
            return cast(Pipeline, base.build(single))

        return LateFusionClassifier(factory, context.active_modalities)

    return ModelSpec(
        name=f"{name}__LATE",
        family=base.family,
        build=build,
        notes="late fusion: one model per modality, probabilities averaged",
    )
