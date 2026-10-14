"""Scikit-learn wrapper exposing the transformer to the shared harness.

This class is what makes a paired comparison against the classical baselines
possible. Wrapped as an estimator, the transformer goes through the same folds,
the same preprocessing pipeline, the same metric functions and the same paired
statistics as every other model — there is no second code path where a protocol
difference could creep in.

Building the graph is the wrinkle. Gene selection is fold-local, so the induced
subgraph, its shortest-path matrix and its positional encoding differ from fold
to fold and cannot be computed once outside the loop. They are therefore built
inside ``fit``, from the genes the preceding transformer selected, and cached on
the fitted estimator for ``predict``. Building them from the full gene universe
instead would leak held-out structure into training.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader, TensorDataset

from mogformer.data.gene_embeddings import align_gene_embeddings
from mogformer.data.omics import MODALITY_ORDER
from mogformer.graph.grn import GRNSignedCache
from mogformer.graph.positional_encoding import GraphPositionalEncoding
from mogformer.graph.spd import compute_shortest_path_matrix
from mogformer.graph.string_graph import StringGraphCache
from mogformer.models.classifier import MultiOmicsGraphClassifier
from mogformer.training.losses import MultiClassFocalLoss, inverse_frequency_weights

logger = logging.getLogger(__name__)


def split_modalities(
    features: np.ndarray,
    n_selected: int,
    active_modalities: Sequence[str] = MODALITY_ORDER,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a transformed matrix into the three modality blocks the model reads.

    A modality left out of ``active_modalities`` carries no information into the
    model: its block is returned as zeros, which the per-feature scaling maps
    to the cohort mean. That is what lets one architecture be scored on a
    modality subset with the same gene selection as the subset's baselines.

    Args:
        features: Transformed features, shape
            ``(n_samples, len(active_modalities) * n_selected)``, blocks in
            modality order.
        n_selected: Genes per block after selection.
        active_modalities: Modalities present in ``features``.

    Returns:
        Expression, copy-number and methylation arrays, each of shape
        ``(n_samples, n_selected)``.

    Raises:
        ValueError: If a modality is unknown, or if the width is not one whole
            block per active modality.
    """
    unknown = [m for m in active_modalities if m not in MODALITY_ORDER]
    if unknown:
        raise ValueError(
            f"unknown modalities {unknown}; available: {list(MODALITY_ORDER)}"
        )
    present = [m for m in MODALITY_ORDER if m in active_modalities]
    expected = len(present) * n_selected
    if features.shape[1] != expected:
        raise ValueError(
            f"expected {expected} columns for {n_selected} selected genes over "
            f"{present}, got {features.shape[1]}"
        )
    blocks = {
        modality: features[:, index * n_selected : (index + 1) * n_selected]
        for index, modality in enumerate(present)
    }
    empty = np.zeros((features.shape[0], n_selected), dtype=features.dtype)
    rna, cnv, methy = (blocks.get(m, empty) for m in MODALITY_ORDER)
    return rna, cnv, methy


class MOGFormerClassifier(BaseEstimator, ClassifierMixin):
    """Train and score the graph transformer through the scikit-learn API.

    Attributes:
        classes_: Class labels seen during ``fit``.
        model_: The trained network.
        spd_: The fold's shortest-path matrix.
        graph_pe_: The fold's positional encodings.
        best_inner_score_: Best inner-validation macro-F1 reached, averaged
            over ``selection_window`` epochs.
        history_: One ``(epoch, mean training loss, inner macro-F1)`` per
            epoch trained, kept so a failed fit can be diagnosed afterwards.
        best_epoch_: Epoch whose weights were kept.
    """

    def __init__(
        self,
        selected_genes: Sequence[str],
        string_cache: StringGraphCache,
        grn_cache: GRNSignedCache | None = None,
        num_classes: int = 5,
        d: int = 128,
        pe_dim: int = 32,
        pe_method: str = "rwpe",
        mini_heads: int = 4,
        global_heads: int = 8,
        global_layers: int = 1,
        dropout: float = 0.2,
        rna_dropout_prob: float = 0.4,
        cnv_dropout_prob: float = 0.2,
        meth_dropout_prob: float = 0.2,
        max_distance: int = 10,
        attention_bias_mode: str = "inside",
        numerical_tokenizer: str = "mlp",
        plr_n_frequencies: int = 16,
        plr_sigma: float = 1.0,
        unimodal_dropout_fill: str = "zero",
        fusion_type: str = "attention",
        gene_id_embedding: bool = False,
        universe_genes: Sequence[str] | None = None,
        pretrained_embeddings: pd.DataFrame | None = None,
        pretrained_emb_adapter_rank: int = 0,
        lambda_gate: bool = False,
        use_grn: bool = False,
        lr: float = 1e-4,
        min_lr: float = 1e-6,
        weight_decay: float = 1e-4,
        max_epochs: int = 300,
        patience: int = 40,
        min_epochs: int = 0,
        selection_window: int = 1,
        batch_size: int = 32,
        inner_val_frac: float = 0.2,
        focal_gamma: float = 2.0,
        class_weight_power: float = 0.5,
        seed: int = 42,
        device: str | None = None,
        active_modalities: Sequence[str] = MODALITY_ORDER,
    ) -> None:
        """Store hyperparameters verbatim for ``sklearn.base.clone``.

        Args:
            selected_genes: Genes chosen by the preceding preprocessing step for
                this fold. Determines the induced graph.
            string_cache: Parsed interaction network over the gene universe.
            grn_cache: Parsed regulatory network, or None.
            num_classes: Number of subtypes.
            d: Token width.
            pe_dim: Width of the positional encoding.
            pe_method: Positional encoding scheme.
            mini_heads: Attention heads in the intra-gene stage.
            global_heads: Attention heads in the inter-gene stage.
            global_layers: Number of inter-gene blocks.
            dropout: Dropout throughout.
            rna_dropout_prob: Probability of hiding expression per gene.
            cnv_dropout_prob: Probability of hiding copy number per gene.
            meth_dropout_prob: Probability of hiding methylation per gene.
            max_distance: Largest hop count represented exactly by the bias.
            attention_bias_mode: Structural bias formulation.
            numerical_tokenizer: Scalar tokenizer.
            plr_n_frequencies: Basis size for the periodic tokenizer.
            plr_sigma: Frequency scale for the periodic tokenizer.
            unimodal_dropout_fill: Fill strategy for dropped modalities.
            fusion_type: Intra-gene fusion layer, attention or gated.
            gene_id_embedding: Attach a per-gene identity embedding, learned
                from ``universe_genes`` or read from
                ``pretrained_embeddings``.
            universe_genes: Every gene in the cohort, defining the identity
                table this fold's genes index into. Required when
                ``gene_id_embedding`` is set without pretrained embeddings.
            pretrained_embeddings: Frozen gene vectors indexed by symbol, or
                None for a learned table.
            pretrained_emb_adapter_rank: Rank of the residual correction
                learned on top of the frozen vectors.
            lambda_gate: Per-head learnable scaling of the distance bias.
            use_grn: Enable the signed regulatory bias.
            lr: AdamW learning rate.
            min_lr: Scheduler floor.
            weight_decay: AdamW weight decay.
            max_epochs: Upper bound on epochs.
            patience: Epochs without improvement before early stopping.
            min_epochs: Epochs trained before early stopping may trigger.
            selection_window: Epochs averaged when comparing inner-validation
                macro-F1 across epochs.
            batch_size: Patients per step.
            inner_val_frac: Fraction of the training fold held out for early
                stopping. The outer fold is never seen during model selection.
            focal_gamma: Focusing exponent of the loss.
            class_weight_power: Exponent on inverse class frequency for the
                loss's class weights; 0.5 is the square root, 1 full inverse.
            seed: Seed applied before training.
            device: Torch device string, or None to pick CUDA when available.
            active_modalities: Modalities the preprocessing step emits. The
                others enter the model as uninformative zero blocks, with
                their unimodal dropout switched off.
        """
        self.selected_genes = selected_genes
        self.string_cache = string_cache
        self.grn_cache = grn_cache
        self.num_classes = num_classes
        self.d = d
        self.pe_dim = pe_dim
        self.pe_method = pe_method
        self.mini_heads = mini_heads
        self.global_heads = global_heads
        self.global_layers = global_layers
        self.dropout = dropout
        self.rna_dropout_prob = rna_dropout_prob
        self.cnv_dropout_prob = cnv_dropout_prob
        self.meth_dropout_prob = meth_dropout_prob
        self.max_distance = max_distance
        self.attention_bias_mode = attention_bias_mode
        self.numerical_tokenizer = numerical_tokenizer
        self.plr_n_frequencies = plr_n_frequencies
        self.plr_sigma = plr_sigma
        self.unimodal_dropout_fill = unimodal_dropout_fill
        self.fusion_type = fusion_type
        self.gene_id_embedding = gene_id_embedding
        self.universe_genes = universe_genes
        self.pretrained_embeddings = pretrained_embeddings
        self.pretrained_emb_adapter_rank = pretrained_emb_adapter_rank
        self.lambda_gate = lambda_gate
        self.use_grn = use_grn
        self.lr = lr
        self.min_lr = min_lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.min_epochs = min_epochs
        self.selection_window = selection_window
        self.batch_size = batch_size
        self.inner_val_frac = inner_val_frac
        self.focal_gamma = focal_gamma
        self.class_weight_power = class_weight_power
        self.seed = seed
        self.device = device
        self.active_modalities = active_modalities

    def _dropout_for(self, modality: str, probability: float) -> float:
        """Return a modality's unimodal dropout, zero when it is not an input.

        Args:
            modality: Modality name.
            probability: Configured dropout probability.

        Returns:
            The probability to use.
        """
        return probability if modality in self.active_modalities else 0.0

    def _resolve_device(self) -> torch.device:
        """Return the device to train on."""
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_graph_tensors(
        self, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Build the fold's graph tensors from the selected genes.

        Args:
            device: Where the tensors should live.

        Returns:
            Tuple of positional encodings, shortest-path matrix, and the signed
            regulatory matrix or None.
        """
        genes = list(self.selected_genes)
        adjacency = torch.as_tensor(
            self.string_cache.induced_adjacency(genes), dtype=torch.float32
        )
        spd = compute_shortest_path_matrix(adjacency, self.max_distance)
        graph_pe = GraphPositionalEncoding(self.pe_dim, self.pe_method)(adjacency)

        grn = None
        if self.grn_cache is not None:
            grn = torch.as_tensor(
                self.grn_cache.induced_signed_adjacency(genes), dtype=torch.float32
            )
        return (
            graph_pe.to(device),
            spd.to(device),
            None if grn is None else grn.to(device),
        )

    def _build_identity_arguments(self) -> dict[str, Any]:
        """Resolve the gene-identity arguments for this fold's gene set.

        Two mutually exclusive sources exist. Pretrained vectors are sliced to
        the fold's genes and frozen; otherwise a learnable table is indexed by
        each selected gene's position in the cohort-wide universe. Either way
        the identity is a property of the gene, never of the patients, so it
        crosses the fold boundary without leaking anything.

        Returns:
            Keyword arguments for :class:`MultiOmicsGraphClassifier`.

        Raises:
            ValueError: If a learnable table is requested without a universe.
        """
        if not self.gene_id_embedding:
            return {"gene_id_embedding": False}

        genes = list(self.selected_genes)
        if self.pretrained_embeddings is not None:
            matrix, covered = align_gene_embeddings(self.pretrained_embeddings, genes)
            self.n_pretrained_covered_ = len(covered)
            return {
                "gene_id_embedding": True,
                "pretrained_gene_emb": torch.as_tensor(matrix, dtype=torch.float32),
                "pretrained_emb_adapter_rank": self.pretrained_emb_adapter_rank,
            }

        if self.universe_genes is None:
            raise ValueError(
                "gene_id_embedding without pretrained_embeddings needs "
                "universe_genes, which indexes the learnable identity table"
            )
        universe = list(self.universe_genes)
        position = {gene: i for i, gene in enumerate(universe)}
        missing = [gene for gene in genes if gene not in position]
        if missing:
            raise ValueError(
                f"{len(missing)} selected gene(s) absent from the universe, "
                f"first few: {missing[:5]}"
            )
        return {
            "gene_id_embedding": True,
            "n_universe": len(universe),
            "gene_ids": [position[gene] for gene in genes],
        }

    def _make_loader(
        self, features: np.ndarray, labels: np.ndarray, shuffle: bool
    ) -> DataLoader:
        """Wrap one split's modality blocks in a data loader.

        Args:
            features: Transformed features for the split.
            labels: Integer labels for the split.
            shuffle: Whether to shuffle between epochs.

        Returns:
            A loader yielding ``(rna, cnv, methy, y)`` batches.
        """
        rna, cnv, methy = split_modalities(
            features, len(self.selected_genes), self.active_modalities
        )
        dataset = TensorDataset(
            torch.as_tensor(rna, dtype=torch.float32),
            torch.as_tensor(cnv, dtype=torch.float32),
            torch.as_tensor(methy, dtype=torch.float32),
            torch.as_tensor(labels, dtype=torch.long),
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=shuffle and len(dataset) > self.batch_size,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> MOGFormerClassifier:
        """Train on one fold, early-stopping on an inner split.

        Args:
            X: Transformed training features, shape
                ``(n_samples, n_active * n_selected)``.
            y: Integer labels, shape ``(n_samples,)``.

        Returns:
            The fitted estimator.

        Raises:
            ValueError: If no genes were selected, which would leave the graph
                empty and the failure downstream unreadable.
        """
        from mogformer.evaluation.metrics import compute_fold_metrics
        from mogformer.training.seed import seed_everything

        if len(self.selected_genes) == 0:
            raise ValueError(
                "selected_genes is empty, so there is no graph to build. Inside "
                "a pipeline this is filled in by GraphAwarePipeline from the "
                "preprocessing step; standalone, pass it explicitly."
            )

        seed_everything(self.seed)
        device = self._resolve_device()
        self.classes_ = np.unique(y)

        graph_pe, spd, grn = self._build_graph_tensors(device)
        self.graph_pe_, self.spd_, self.grn_ = graph_pe, spd, grn

        train_idx, inner_idx = train_test_split(
            np.arange(len(y)),
            test_size=self.inner_val_frac,
            stratify=y,
            random_state=self.seed,
        )
        train_loader = self._make_loader(X[train_idx], y[train_idx], shuffle=True)
        inner_loader = self._make_loader(X[inner_idx], y[inner_idx], shuffle=False)

        model = MultiOmicsGraphClassifier(
            num_classes=self.num_classes,
            d=self.d,
            pe_dim=self.pe_dim,
            mini_heads=self.mini_heads,
            global_heads=self.global_heads,
            global_layers=self.global_layers,
            dropout=self.dropout,
            rna_dropout_prob=self._dropout_for("rna", self.rna_dropout_prob),
            cnv_dropout_prob=self._dropout_for("cnv", self.cnv_dropout_prob),
            meth_dropout_prob=self._dropout_for("methy", self.meth_dropout_prob),
            max_distance=self.max_distance,
            attention_bias_mode=self.attention_bias_mode,
            numerical_tokenizer=self.numerical_tokenizer,
            plr_n_frequencies=self.plr_n_frequencies,
            plr_sigma=self.plr_sigma,
            unimodal_dropout_fill=self.unimodal_dropout_fill,
            fusion_type=self.fusion_type,
            lambda_gate=self.lambda_gate,
            use_grn=self.use_grn,
            **self._build_identity_arguments(),
        ).to(device)

        weights = torch.as_tensor(
            inverse_frequency_weights(
                y[train_idx], self.num_classes, self.class_weight_power
            ),
            dtype=torch.float32,
        ).to(device)
        criterion = MultiClassFocalLoss(alpha=weights, gamma=self.focal_gamma)
        optimiser = torch.optim.AdamW(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=self.max_epochs, eta_min=self.min_lr
        )

        best_score = -np.inf
        best_state: dict[str, Any] | None = None
        epochs_without_improvement = 0
        self.history_: list[tuple[int, float, float]] = []
        self.best_epoch_ = -1

        for epoch in range(self.max_epochs):
            model.train()
            losses = []
            for rna, cnv, methy, labels in train_loader:
                optimiser.zero_grad()
                logits = model(
                    rna.to(device),
                    cnv.to(device),
                    methy.to(device),
                    graph_pe,
                    spd,
                    grn,
                )["logits"]
                loss = criterion(logits, labels.to(device))
                losses.append(float(loss.detach()))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()
            scheduler.step()

            self.model_ = model
            predictions, truths = self._infer(inner_loader, device)
            score = compute_fold_metrics(
                truths,
                predictions.argmax(axis=1),
                predictions,
                list(range(self.num_classes)),
                [str(i) for i in range(self.num_classes)],
            )["macro_f1"]
            self.history_.append((epoch, float(np.mean(losses)), float(score)))
            window = [entry[2] for entry in self.history_[-self.selection_window :]]
            smoothed = float(np.mean(window))

            if smoothed > best_score:
                best_score = smoothed
                self.best_epoch_ = epoch
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                epochs_without_improvement = 0
            elif epoch >= self.min_epochs:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    logger.info("early stop at epoch %d", epoch)
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        self.model_ = model
        self.best_inner_score_ = float(best_score)
        return self

    def _infer(
        self, loader: DataLoader, device: torch.device
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run the fitted model over a loader without gradients.

        Uses the graph tensors cached by ``fit``, so held-out patients are
        scored against the same fold-local graph the model was trained on.

        Args:
            loader: Batches to score.
            device: Device the graph tensors live on.

        Returns:
            Tuple of predicted probabilities and the labels the loader carried.
        """
        self.model_.eval()
        probabilities, truths = [], []
        with torch.no_grad():
            for rna, cnv, methy, labels in loader:
                logits = self.model_(
                    rna.to(device),
                    cnv.to(device),
                    methy.to(device),
                    self.graph_pe_,
                    self.spd_,
                    self.grn_,
                )["logits"]
                probabilities.append(torch.softmax(logits, dim=-1).cpu().numpy())
                truths.append(labels.numpy())
        return np.vstack(probabilities), np.concatenate(truths)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities for held-out patients.

        Args:
            X: Transformed features, shape ``(n_samples, n_active * n_selected)``.

        Returns:
            Probabilities of shape ``(n_samples, num_classes)``.
        """
        device = self._resolve_device()
        loader = self._make_loader(X, np.zeros(len(X), dtype=np.int64), shuffle=False)
        probabilities, _ = self._infer(loader, device)
        return probabilities

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the most likely class for each held-out patient.

        Args:
            X: Transformed features, shape ``(n_samples, n_active * n_selected)``.

        Returns:
            Integer labels of shape ``(n_samples,)``.
        """
        return self.predict_proba(X).argmax(axis=1)


class GraphAwarePipeline(Pipeline):
    """A pipeline that tells its final estimator which genes it selected.

    A plain :class:`~sklearn.pipeline.Pipeline` cannot express this dependency.
    Gene selection happens inside the fitted preprocessing step, but the
    transformer's graph — its induced adjacency, shortest paths and positional
    encodings — is a function of exactly those genes, and nothing downstream of
    a pipeline step ordinarily learns anything about it beyond the column
    count. Passing the universe instead would leak held-out structure into
    training, and hardcoding a gene list would silently decouple the graph from
    the data.

    So this subclass closes the loop: after the preprocessing step is fit, it
    reads the selected symbols off it and writes them onto the final estimator
    before that estimator sees a single row.
    """

    def _fit(self, *args: Any, **kwargs: Any) -> Any:
        """Fit the transformers, then hand their gene selection downstream.

        Everything is forwarded untouched, because the private hook this
        overrides has gained arguments across scikit-learn releases and the
        only thing that matters here is *when* it runs: after the transformers
        are fit, before the final estimator sees anything.

        Args:
            *args: Whatever the base class was called with.
            **kwargs: Whatever the base class was called with.

        Returns:
            The transformed training features.
        """
        transformed = super()._fit(*args, **kwargs)
        selector = self.steps[0][1]
        estimator = self.steps[-1][1]
        if hasattr(selector, "get_selected_gene_names") and hasattr(
            estimator, "selected_genes"
        ):
            estimator.set_params(selected_genes=selector.get_selected_gene_names())
        return transformed
