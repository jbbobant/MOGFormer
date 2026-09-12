"""Typed configuration objects, loaded from YAML.

Every knob the project has reaches the code through one of these dataclasses.
Nothing about a cohort, a file layout or a hyperparameter is a literal buried in
a function, because the one time a cohort filter *was* a literal it silently
desynchronised the baselines from the transformer for months.

A run writes its resolved configuration next to its results, so a result can
always be traced back to the settings that produced it.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

from mogformer.data.omics import MODALITY_ORDER

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class DataConfig:
    """Where the cohort lives and which patients it contains.

    Attributes:
        raw_dir: Directory holding the four input matrices.
        rna_file: Expression matrix filename, genes by patients.
        cnv_file: Copy-number matrix filename.
        methy_file: Methylation matrix filename.
        clin_file: Clinical table filename.
        label_col: Column of the clinical table holding the class label.
        exclude_classes: Classes dropped from the cohort. Recording this
            explicitly is what keeps a four-class run comparable to a five-class
            one rather than a separate experiment.
        curated_genes_file: Genes force-included after variance ranking, or None.
    """

    raw_dir: str = "data/raw"
    rna_file: str = "data_rna_seq_v2_rsem.csv"
    cnv_file: str = "data_cnv.csv"
    methy_file: str = "data_methylation450.csv"
    clin_file: str = "data_clinical.csv"
    label_col: str = "SUBTYPE"
    exclude_classes: tuple[str, ...] = ()
    curated_genes_file: str | None = None


@dataclass
class FoldConfig:
    """The shared cross-validation partition.

    Attributes:
        path: Where the partition is persisted. Every model reads this file, and
            that is the whole point of it existing.
        n_splits: Folds per repeat. Five rather than ten keeps roughly seven of
            the rarest class in each held-out fold.
        n_repeats: Number of repeats; estimates is the product with n_splits.
        seed: Seed for the repeated splitter.
        n_repeats_used: Use only the first this-many repeats, for cheap partial
            runs. None uses every repeat.
        max_folds: Use only the first this-many folds per repeat. None uses all.
    """

    path: str = "results/folds.json"
    n_splits: int = 5
    n_repeats: int = 5
    seed: int = 42
    n_repeats_used: int | None = None
    max_folds: int | None = None


@dataclass
class PreprocessConfig:
    """Per-fold feature selection and scaling.

    Attributes:
        top_k: Size of the variance-ranked gene pool.
        active_modalities: Modalities used, driving both selection and output.
        mad_on_log_rna: Rank expression by the deviation of its log.
        impute: Replace missing values with training-fold medians.
    """

    top_k: int = 250
    active_modalities: tuple[str, ...] = MODALITY_ORDER
    mad_on_log_rna: bool = False
    impute: bool = True


@dataclass
class GraphConfig:
    """The biological priors and their derived tensors.

    Attributes:
        ppi_file: STRING protein links file.
        alias_file: STRING protein aliases file.
        grn_file: CollecTRI signed edges file, or None to disable.
        string_threshold: Minimum STRING combined score.
        max_distance: Largest hop count represented exactly by the bias.
        pe_dim: Width of the positional encoding.
        pe_method: One of
            :data:`~mogformer.graph.positional_encoding.PE_METHODS`.
        use_grn: Enable the signed regulatory attention bias.
    """

    ppi_file: str = "data/clean/9606.protein.links.v12.0.txt"
    alias_file: str = "data/clean/9606.protein.aliases.v12.0.txt"
    grn_file: str | None = None
    string_threshold: int = 400
    max_distance: int = 10
    pe_dim: int = 32
    pe_method: str = "rwpe"
    use_grn: bool = False


@dataclass
class ModelConfig:
    """Architecture of the transformer.

    Attributes:
        d: Token width.
        mini_heads: Attention heads in the intra-gene stage.
        global_heads: Attention heads in the inter-gene stage.
        global_layers: Number of inter-gene blocks.
        dropout: Dropout throughout.
        rna_dropout_prob: Probability of hiding expression per gene.
        cnv_dropout_prob: Probability of hiding copy number per gene.
        meth_dropout_prob: Probability of hiding methylation per gene.
        attention_bias_mode: One of
            :data:`~mogformer.models.layers.structural_attention.ATTENTION_BIAS_MODES`.
        numerical_tokenizer: One of
            :data:`~mogformer.models.layers.modality_lifting.NUMERICAL_TOKENIZERS`.
        plr_n_frequencies: Basis size for the periodic tokenizer.
        plr_sigma: Frequency scale for the periodic tokenizer.
        unimodal_dropout_fill: One of
            :data:`~mogformer.models.layers.modality_dropout.DROPOUT_FILLS`.
        gene_id_embedding: Add per-gene identity embeddings.
        pretrained_emb_file: Pretrained gene embeddings, or None.
        pretrained_emb_adapter_rank: Adapter rank for the pretrained path.
        lambda_gate: Per-head learnable scaling of the distance bias.
        fusion_type: Intra-gene fusion layer, one of
            :data:`~mogformer.models.ssl.FUSION_TYPES`.
    """

    d: int = 128
    mini_heads: int = 4
    global_heads: int = 8
    global_layers: int = 1
    dropout: float = 0.2
    rna_dropout_prob: float = 0.4
    cnv_dropout_prob: float = 0.2
    meth_dropout_prob: float = 0.2
    attention_bias_mode: str = "inside"
    numerical_tokenizer: str = "mlp"
    plr_n_frequencies: int = 16
    plr_sigma: float = 1.0
    unimodal_dropout_fill: str = "zero"
    gene_id_embedding: bool = False
    pretrained_emb_file: str | None = None
    pretrained_emb_adapter_rank: int = 0
    lambda_gate: bool = False
    fusion_type: str = "attention"


@dataclass
class TrainConfig:
    """Optimisation and early stopping.

    Attributes:
        lr: AdamW learning rate.
        min_lr: Floor for the scheduler.
        weight_decay: AdamW weight decay.
        max_epochs: Upper bound on training epochs.
        patience: Epochs without inner-validation improvement before stopping.
        batch_size: Patients per step.
        inner_val_frac: Fraction of the training fold held out for early
            stopping. The outer fold is never seen during model selection.
        focal_gamma: Focusing exponent of the classification loss.
        log_every: Epoch interval for progress logging.
        seed: Seed applied before training.
    """

    lr: float = 1e-4
    min_lr: float = 1e-6
    weight_decay: float = 1e-4
    max_epochs: int = 300
    patience: int = 40
    batch_size: int = 32
    inner_val_frac: float = 0.2
    focal_gamma: float = 2.0
    log_every: int = 1
    seed: int = 42


@dataclass
class ExperimentConfig:
    """A complete run: cohort, folds, preprocessing, priors, model, training.

    Attributes:
        name: Run identifier, used as the results subdirectory.
        results_dir: Root under which run outputs are written.
        data: Cohort configuration.
        folds: Partition configuration.
        preprocess: Per-fold preprocessing configuration.
        graph: Biological prior configuration.
        model: Architecture configuration.
        train: Optimisation configuration.
    """

    name: str = "baseline"
    results_dir: str = "results"
    data: DataConfig = field(default_factory=DataConfig)
    folds: FoldConfig = field(default_factory=FoldConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @property
    def output_dir(self) -> Path:
        """Return the directory this run writes into."""
        return Path(self.results_dir) / self.name


#: Nested configuration sections of :class:`ExperimentConfig`, by field name.
_SECTIONS: dict[str, type] = {
    "data": DataConfig,
    "folds": FoldConfig,
    "preprocess": PreprocessConfig,
    "graph": GraphConfig,
    "model": ModelConfig,
    "train": TrainConfig,
}


def _build_section(cls: type[T], values: dict[str, Any], path: str) -> T:
    """Construct one configuration section from a mapping.

    Args:
        cls: Section dataclass to construct.
        values: Mapping of field name to value.
        path: Section name, used in error messages.

    Returns:
        The constructed section.

    Raises:
        ValueError: If ``values`` contains a key the section does not declare.
            Unknown keys are rejected rather than ignored, because a silently
            dropped setting produces a run that did not do what its file says.
    """
    declared = {f.name: f for f in fields(cls)}  # type: ignore[arg-type]
    unknown = set(values) - set(declared)
    if unknown:
        raise ValueError(
            f"unknown configuration key(s) under {path!r}: {sorted(unknown)}; "
            f"expected any of {sorted(declared)}"
        )

    kwargs: dict[str, Any] = {}
    for name, value in values.items():
        # YAML has no tuple literal, so a list feeding a tuple field is coerced.
        if isinstance(value, list) and "tuple" in str(declared[name].type):
            kwargs[name] = tuple(value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> ExperimentConfig:
    """Read an experiment configuration from a YAML file.

    Args:
        path: YAML file. Absent sections fall back to their dataclass defaults.

    Returns:
        The resolved configuration.

    Raises:
        ValueError: If the file declares a key no dataclass has.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a mapping at the top level")

    top_level = {k: v for k, v in raw.items() if k not in _SECTIONS}
    unknown = set(top_level) - {f.name for f in fields(ExperimentConfig)}
    if unknown:
        raise ValueError(
            f"unknown configuration key(s) at the top level: {sorted(unknown)}"
        )

    sections: dict[str, Any] = {
        name: _build_section(cls, raw.get(name) or {}, name)
        for name, cls in _SECTIONS.items()
    }
    config = ExperimentConfig(**top_level, **sections)
    logger.info("loaded configuration %r from %s", config.name, path)
    return config


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    """Write a resolved configuration beside its results.

    Args:
        config: Configuration to record.
        path: Destination YAML file; parent directories are created as needed.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(dataclasses.asdict(config), sort_keys=False),
        encoding="utf-8",
    )
    logger.info("wrote resolved configuration to %s", destination)
