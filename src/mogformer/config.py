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
        cohort_file: A cohort archive written by
            :func:`~mogformer.data.omics.save_omics_bundle`, read instead of the
            raw matrices when set.
        ploidy_file: Per-patient ploidy table (``patient_id``, ``ploidy``),
            estimated over the full absolute copy-number matrix; required by
            the ``ploidy_relative`` copy-number scale.
        cohort_rule: Which patients the cohort contains and why, in words.
            Recorded in every provenance record, so a cohort is never implied.
        batch_file: Per-patient table (patients in the first column) holding
            the processing batch, relative to ``raw_dir``; read when
            ``preprocess.batch_correction`` is set.
        batch_column: Column of ``batch_file`` holding the batch.
    """

    raw_dir: str = "data/raw"
    rna_file: str = "data_rna_seq_v2_rsem.csv"
    cnv_file: str = "data_cnv.csv"
    methy_file: str = "data_methylation450.csv"
    clin_file: str = "data_clinical.csv"
    label_col: str = "SUBTYPE"
    exclude_classes: tuple[str, ...] = ()
    curated_genes_file: str | None = None
    cohort_file: str | None = None
    ploidy_file: str | None = None
    cohort_rule: str | None = None
    batch_file: str | None = None
    batch_column: str | None = None


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
        rna_scale: ``"counts"`` when expression arrives as non-negative
            counts, which are log-transformed; ``"log"`` when it is already on
            a log or standardised scale, which is only standardised.
        cnv_representation: Scale copy number is presented on, one of
            :data:`~mogformer.data.omics.CNV_REPRESENTATIONS`.
        clip_z: Clip every standardised value to ``[-clip_z, clip_z]``, or
            None to leave the tails unbounded.
        batch_correction: Remove processing-batch effects before anything
            else, one of
            :data:`~mogformer.data.batch_correction.BATCH_CORRECTIONS`, or
            None. Fitted on the fitting patients only.
        batch_modalities: Omics the batch correction is applied to.
    """

    top_k: int = 250
    active_modalities: tuple[str, ...] = MODALITY_ORDER
    mad_on_log_rna: bool = False
    impute: bool = True
    rna_scale: str = "counts"
    cnv_representation: str = "absolute"
    clip_z: float | None = None
    batch_correction: str | None = None
    batch_modalities: tuple[str, ...] = ("rna", "methy")


@dataclass
class GraphConfig:
    """The biological priors and their derived tensors.

    Attributes:
        ppi_file: STRING protein links file.
        alias_file: STRING protein aliases file.
        grn_file: CollecTRI signed edges file, or None to disable.
        edges_file: A gene-symbol edge list already reduced to this cohort's
            universe. When set, ``ppi_file`` and ``alias_file`` are not read —
            the 800 MB STRING pair is only needed to produce this file once.
        string_threshold: Minimum STRING combined score.
        max_distance: Largest hop count represented exactly by the bias.
        pe_dim: Width of the positional encoding.
        pe_method: One of
            :data:`~mogformer.graph.positional_encoding.PE_METHODS`.
        use_grn: Enable the signed regulatory attention bias.
        weighted_pe: Weight random-walk positional encodings by edge
            confidence, when the edge list carries a score.
    """

    ppi_file: str = "data/clean/9606.protein.links.v12.0.txt"
    alias_file: str = "data/clean/9606.protein.aliases.v12.0.txt"
    grn_file: str | None = None
    edges_file: str | None = None
    string_threshold: int = 400
    max_distance: int = 10
    pe_dim: int = 32
    pe_method: str = "rwpe"
    use_grn: bool = False
    weighted_pe: bool = False


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
        pretrained_emb_file: Pretrained gene embeddings, as either a
            headerless matrix or an ``.npz`` bundle, or None.
        pretrained_emb_symbol_file: Row-aligned gene symbols, required only
            when ``pretrained_emb_file`` is a headerless matrix.
        pretrained_emb_adapter_rank: Adapter rank for the pretrained path.
        lambda_gate: Per-head learnable scaling of the distance bias.
        fusion_type: Intra-gene fusion layer, one of
            :data:`~mogformer.models.ssl.FUSION_TYPES`.
        mask_gene_frac: Self-supervision: fraction of genes hidden entirely.
        mask_modality_frac: Self-supervision: probability of hiding one
            modality of a gene.
        mask_modality_weights: Self-supervision: relative weights over
            ``(rna, cnv, methy)`` when choosing the modality to hide.
        global_offset: Self-supervision: let the decoder's global head add a
            patient-level offset shared by every gene and modality.
        isolated_token: Give genes without an edge in the interaction graph a
            learned positional code of their own.
        omics_mask_prob: Self-supervision: probability, per patient, of
            hiding one whole omics.
        neighbourhood_mask_share: Self-supervision: share of whole-gene masks
            drawn as a gene plus its one-hop interaction neighbours.
        identity_modulation: Self-supervision: gene identity scales and
            shifts each value token (FiLM), not only sits beside it.
        n_summary_tokens: Self-supervision: summary tokens ahead of the
            genes; the embedding is their concatenation.
        global_head: Self-supervision: decoder global head, one of
            :data:`~mogformer.models.layers.decoder.GLOBAL_HEADS`.
        fusion_residual: Self-supervision: carry the unfused omics tokens
            past the gated fusion.
        program_source: Self-supervision: where gene programs come from, one
            of :data:`~mogformer.data.gene_programs.PROGRAM_SOURCES`, or None
            for the summary-token embedding.
        n_programs: Self-supervision: number of gene programs; each gets a
            scoped read-out and scopes the decoder's global head.
        program_width: Self-supervision: width of each program embedding; the
            patient embedding is ``n_programs * program_width`` wide.
        program_layers: Self-supervision: read-out layers of the program
            queries.
        program_input: Self-supervision: gene states the program queries
            read, ``"final"`` (after the inter-gene transformer) or
            ``"fused"`` (each gene's own token, before any mixing).
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
    pretrained_emb_symbol_file: str | None = None
    pretrained_emb_adapter_rank: int = 0
    lambda_gate: bool = False
    fusion_type: str = "attention"
    mask_gene_frac: float = 0.15
    mask_modality_frac: float = 0.15
    mask_modality_weights: tuple[float, ...] = (2.0, 1.0, 1.0)
    global_offset: bool = True
    isolated_token: bool = False
    omics_mask_prob: float = 0.0
    program_source: str | None = None
    n_programs: int = 0
    program_width: int = 32
    program_layers: int = 2
    program_input: str = "final"
    neighbourhood_mask_share: float = 0.0
    identity_modulation: bool = False
    n_summary_tokens: int = 1
    global_head: str = "film"
    fusion_residual: bool = False


@dataclass
class TrainConfig:
    """Optimisation and early stopping.

    Attributes:
        lr: AdamW learning rate.
        min_lr: Floor for the scheduler.
        weight_decay: AdamW weight decay.
        max_epochs: Upper bound on training epochs.
        patience: Epochs without inner-validation improvement before stopping.
        min_epochs: Epochs trained before early stopping may trigger. Training
            can sit on a plateau for tens of epochs before it improves; stopping
            during it keeps a model that never learned.
        selection_window: Epochs over which inner-validation macro-F1 is
            averaged before comparing epochs. A small inner split makes the
            per-epoch score noisy, and selecting on single-epoch spikes keeps a
            lucky epoch rather than a good one. One means no averaging.
        batch_size: Patients per step.
        inner_val_frac: Fraction of the training fold held out for early
            stopping. The outer fold is never seen during model selection.
        focal_gamma: Focusing exponent of the classification loss.
        class_weight_power: Exponent on inverse class frequency for the loss's
            class weights: 0.5 (square root) by default, 1 for full inverse
            frequency, 0 for equal weights.
        log_every: Epoch interval for progress logging.
        seed: Seed applied before training.
        gate_lr_multiplier: Learning-rate multiplier of the attention-bias
            gates relative to every other weight.
        warmup_epochs: Epochs of linear learning-rate warm-up.
        lr_schedule: ``"constant"``, or ``"cosine"`` decay from ``lr`` to
            ``min_lr`` over the epochs after warm-up.
        per_modality_loss: Self-supervision: average the reconstruction loss
            within each modality, then sum, so each modality's share of the
            gradient does not depend on how often it is masked.
        monitor_frac: Self-supervision: share of patients used only to decide
            when to stop.
        heldout_frac: Self-supervision: share of patients never trained on and
            never used for stopping, kept to measure reconstruction.
        split_seed: Self-supervision: seed of the held-out and monitoring
            splits and of the fixed monitoring masks, kept apart from ``seed``
            so that encoders trained with different seeds share their splits.
        n_readout_masks: Self-supervision: random masks averaged over for the
            masked summary read-out.
        lambda_global: Self-supervision: weight of the decoder's global-head
            loss (the local head's is 1).
        decorrelation_penalty: Self-supervision: weight of the embedding
            decorrelation penalty (0 disables it).
        uniformity_weight: Self-supervision: weight of the uniformity term on
            the normalised summary (0 disables it).
        vicreg_weight: Self-supervision: weight of VICReg's variance and
            covariance terms on the expanded summary (0 disables them).
        memory_bank_size: Self-supervision: recent summaries the two
            regularisers read beside the batch.
        expander_width: Self-supervision: width of the VICReg expander.
        selection: Self-supervision: ``"loss"`` keeps the epoch with the best
            monitoring loss; ``"loss_then_rank"`` keeps, among the epochs
            within ``selection_tolerance`` of it, the one with the highest
            RankMe.
        selection_tolerance: Self-supervision: relative loss tolerance of
            ``"loss_then_rank"``.
        contrastive_weight: Self-supervision: weight of the cross-omics
            InfoNCE loss between two views of each patient that each lack a
            different omics (0 disables the two-view step).
        contrastive_temperature: Self-supervision: InfoNCE temperature.
        projection_width: Self-supervision: output width of the contrastive
            projection head.
        two_views: Self-supervision: train on the two cross-omics views even
            without the contrastive loss (the control for it).
        view_omics_prob: Self-supervision: probability, per patient and view,
            that a view hides a whole omics (the two views then hide
            different ones); 1.0 hides one in every view.
    """

    lr: float = 1e-4
    min_lr: float = 1e-6
    weight_decay: float = 1e-4
    max_epochs: int = 300
    patience: int = 40
    min_epochs: int = 0
    selection_window: int = 1
    batch_size: int = 32
    inner_val_frac: float = 0.2
    focal_gamma: float = 2.0
    class_weight_power: float = 0.5
    log_every: int = 1
    seed: int = 42
    gate_lr_multiplier: float = 10.0
    warmup_epochs: int = 0
    lr_schedule: str = "constant"
    per_modality_loss: bool = False
    monitor_frac: float = 0.12
    heldout_frac: float = 0.0
    split_seed: int = 42
    n_readout_masks: int = 10
    lambda_global: float = 1.0
    decorrelation_penalty: float = 0.0
    uniformity_weight: float = 0.0
    vicreg_weight: float = 0.0
    memory_bank_size: int = 512
    expander_width: int = 512
    selection: str = "loss"
    selection_tolerance: float = 0.01
    contrastive_weight: float = 0.0
    contrastive_temperature: float = 0.3
    projection_width: int = 128
    two_views: bool = False
    view_omics_prob: float = 1.0


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
