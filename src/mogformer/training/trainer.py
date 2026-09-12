"""Training loop for the masked multi-modal reconstruction objective.

Pretraining is monitored on more than its loss. A reconstruction loss can fall
while the representation quietly collapses onto a single direction, so the
participation ratio is tracked every epoch alongside it, and per-modality losses
are reported against a predict-the-mean baseline — a head that beats the loss
but not that baseline has learned the marginal distribution, not the biology.

The structural-bias gates get their own optimiser group. They are a handful of
scalars competing with millions of weights, and at the shared learning rate they
barely move within the epoch budget; a much larger rate and no weight decay lets
them actually find their scale, which is the point of having them.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import cast

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from mogformer.data.omics import MODALITY_ORDER
from mogformer.models.layers.global_transformer import GlobalGraphTransformer
from mogformer.models.layers.structural_attention import StructuralAttentionBlock
from mogformer.training.losses import masked_huber_dual, participation_ratio

logger = logging.getLogger(__name__)

#: Learning-rate multiplier for the structural-bias gates. See the module note.
GATE_LR_MULTIPLIER = 1000.0


@dataclass
class ValidationScores:
    """One epoch's monitoring diagnostics.

    Attributes:
        val: Combined monitoring loss.
        loss_global: Global-head component.
        loss_local: Local-head component.
        participation_ratio: Effective embedding dimensionality.
        per_modality: Local-head loss per modality, in modality order.
        baseline: Predict-the-mean loss per modality.
    """

    val: float
    loss_global: float
    loss_local: float
    participation_ratio: float
    per_modality: list[float]
    baseline: list[float]


@dataclass
class TrainingHistory:
    """Per-epoch diagnostics collected during pretraining.

    Attributes:
        train: Training loss per epoch.
        val: Monitoring loss per epoch.
        loss_global: Global-head component of the monitoring loss.
        loss_local: Local-head component of the monitoring loss.
        participation_ratio: Effective dimensionality of the embedding.
        per_modality: Local-head loss per modality, in modality order.
        baseline: Predict-the-mean loss per modality, for comparison.
        grn_activation: Summed magnitude of the activating regulatory bias.
        grn_repression: Summed magnitude of the repressing regulatory bias.
    """

    train: list[float] = field(default_factory=list)
    val: list[float] = field(default_factory=list)
    loss_global: list[float] = field(default_factory=list)
    loss_local: list[float] = field(default_factory=list)
    participation_ratio: list[float] = field(default_factory=list)
    per_modality: list[list[float]] = field(default_factory=list)
    baseline: list[list[float]] = field(default_factory=list)
    grn_activation: list[float] = field(default_factory=list)
    grn_repression: list[float] = field(default_factory=list)


class MaskedReconstructionTrainer:
    """Train the self-supervised encoder and monitor it for collapse.

    Attributes:
        model: The encoder being trained.
        history: Per-epoch diagnostics.
        best_val: Best monitoring loss reached.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        spd_matrix: torch.Tensor,
        graph_pe: torch.Tensor,
        grn_matrix: torch.Tensor | None = None,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        lambda_global: float = 1.0,
        lambda_local: float = 1.0,
        delta: float = 1.0,
    ) -> None:
        """Set up the optimiser with a separate group for the bias gates.

        Args:
            model: Encoder to train.
            train_loader: Batches used for the backward pass.
            val_loader: Batches used only for monitoring and early stopping.
            device: Where to train.
            spd_matrix: Integer gene distances for this gene set.
            graph_pe: Positional encodings for this gene set.
            grn_matrix: Signed regulatory adjacency, or None.
            lr: Base AdamW learning rate.
            weight_decay: AdamW weight decay, not applied to the gates.
            lambda_global: Weight of the global head's reconstruction term.
            lambda_local: Weight of the local head's term.
            delta: Huber transition point, in standardised units.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.graph_pe = graph_pe
        self.spd = spd_matrix
        self.grn = grn_matrix
        self.lambda_global = lambda_global
        self.lambda_local = lambda_local
        self.delta = delta

        gate_params = [
            p for name, p in model.named_parameters() if "lambda_raw" in name
        ]
        other_params = [
            p for name, p in model.named_parameters() if "lambda_raw" not in name
        ]
        self.optimizer = torch.optim.AdamW(
            [
                {"params": other_params, "weight_decay": weight_decay},
                {
                    "params": gate_params,
                    "weight_decay": 0.0,
                    "lr": lr * GATE_LR_MULTIPLIER,
                },
            ],
            lr=lr,
        )

        self.best_val = float("inf")
        self.best_state: dict[str, torch.Tensor] | None = None
        self.history = TrainingHistory()

    def _to_device(
        self, batch: Iterable[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Move one batch's three modality tensors onto the training device."""
        rna, cnv, methy = tuple(batch)[:3]
        return rna.to(self.device), cnv.to(self.device), methy.to(self.device)

    def _forward(
        self, rna: torch.Tensor, cnv: torch.Tensor, methy: torch.Tensor, mask: bool
    ) -> dict[str, torch.Tensor]:
        """Run the encoder with this trainer's graph tensors."""
        return self.model(rna, cnv, methy, self.graph_pe, self.spd, self.grn, mask=mask)

    def train_epoch(self) -> float:
        """Run one pass over the training batches.

        Returns:
            Mean training loss over the epoch.
        """
        self.model.train()
        total, n_batches = 0.0, 0
        for batch in self.train_loader:
            rna, cnv, methy = self._to_device(batch)
            self.optimizer.zero_grad()
            out = self._forward(rna, cnv, methy, mask=True)
            loss, _, _ = masked_huber_dual(
                out["xhat_g"],
                out["xhat_l"],
                out["targets"],
                out["mask_bool"],
                self.lambda_global,
                self.lambda_local,
                self.delta,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total += loss.item()
            n_batches += 1
        return total / max(n_batches, 1)

    @torch.no_grad()
    def validate_epoch(self) -> ValidationScores:
        """Score the monitoring split and collect collapse diagnostics.

        Returns:
            Mapping with the monitoring loss, both head components, the
            participation ratio, and per-modality losses beside their
            predict-the-mean baselines.
        """
        self.model.eval()
        total = global_total = local_total = 0.0
        n_batches = 0
        embeddings = []

        modality_loss = [0.0] * len(MODALITY_ORDER)
        baseline_loss = [0.0] * len(MODALITY_ORDER)
        modality_count = [0] * len(MODALITY_ORDER)

        for batch in self.val_loader:
            rna, cnv, methy = self._to_device(batch)
            out = self._forward(rna, cnv, methy, mask=True)
            loss, loss_global, loss_local = masked_huber_dual(
                out["xhat_g"],
                out["xhat_l"],
                out["targets"],
                out["mask_bool"],
                self.lambda_global,
                self.lambda_local,
                self.delta,
            )
            total += loss.item()
            global_total += loss_global.item()
            local_total += loss_local.item()
            n_batches += 1
            embeddings.append(out["c"].cpu())

            mask_bool, targets, local = (
                out["mask_bool"],
                out["targets"],
                out["xhat_l"],
            )
            for k in range(len(MODALITY_ORDER)):
                selected = mask_bool[..., k]
                if not selected.any():
                    continue
                true_values = targets[..., k][selected]
                modality_loss[k] += F.huber_loss(
                    local[..., k][selected],
                    true_values,
                    delta=self.delta,
                    reduction="sum",
                ).item()
                # Standardised targets are zero-mean, so predicting zero is the
                # predict-the-mean baseline this head has to beat.
                baseline_loss[k] += F.huber_loss(
                    torch.zeros_like(true_values),
                    true_values,
                    delta=self.delta,
                    reduction="sum",
                ).item()
                modality_count[k] += int(selected.sum().item())

        return ValidationScores(
            val=total / max(n_batches, 1),
            loss_global=global_total / max(n_batches, 1),
            loss_local=local_total / max(n_batches, 1),
            participation_ratio=participation_ratio(torch.cat(embeddings, 0)),
            per_modality=[
                modality_loss[k] / modality_count[k]
                if modality_count[k]
                else float("nan")
                for k in range(len(MODALITY_ORDER))
            ],
            baseline=[
                baseline_loss[k] / modality_count[k]
                if modality_count[k]
                else float("nan")
                for k in range(len(MODALITY_ORDER))
            ],
        )

    def _record_gate_magnitudes(self) -> None:
        """Append the regulatory bias magnitudes of the first block.

        Tracked because a bias that never leaves zero means the regulatory
        prior is contributing nothing, which is a result rather than a bug.
        """
        transformer = cast(GlobalGraphTransformer, self.model.global_transformer)
        attention = cast(StructuralAttentionBlock, transformer.layers[0]).attn
        self.history.grn_activation.append(
            float(attention.b_grn_activation.abs().sum().item())
        )
        self.history.grn_repression.append(
            float(attention.b_grn_repression.abs().sum().item())
        )

    def fit_early(self, max_epochs: int, patience: int, log_every: int = 1) -> float:
        """Train until the monitoring loss stops improving.

        The best state is restored before returning, so the trainer always ends
        holding the model that scored best rather than the last one.

        Args:
            max_epochs: Upper bound on epochs.
            patience: Epochs without improvement before stopping.
            log_every: Epoch interval for progress logging.

        Returns:
            The best monitoring loss reached.
        """
        epochs_without_improvement = 0

        for epoch in range(1, max_epochs + 1):
            train_loss = self.train_epoch()
            scores = self.validate_epoch()

            self.history.train.append(train_loss)
            self.history.val.append(scores.val)
            self.history.loss_global.append(scores.loss_global)
            self.history.loss_local.append(scores.loss_local)
            self.history.participation_ratio.append(scores.participation_ratio)
            self.history.per_modality.append(list(scores.per_modality))
            self.history.baseline.append(list(scores.baseline))
            self._record_gate_magnitudes()

            if epoch % log_every == 0:
                per_modality = scores.per_modality
                baseline = scores.baseline
                logger.info(
                    "epoch %03d | train %.4f | val %.4f (global %.4f local %.4f) "
                    "| participation %.2f | per-modality %s vs baseline %s",
                    epoch,
                    train_loss,
                    scores.val,
                    scores.loss_global,
                    scores.loss_local,
                    scores.participation_ratio,
                    "/".join(f"{v:.3f}" for v in per_modality),
                    "/".join(f"{v:.3f}" for v in baseline),
                )

            if scores.val < self.best_val - 1e-5:
                self.best_val = scores.val
                self.best_state = copy.deepcopy(self.model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    logger.info(
                        "early stop at epoch %d; best monitoring loss %.4f",
                        epoch,
                        self.best_val,
                    )
                    break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        return self.best_val

    @torch.no_grad()
    def harvest_gates(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Collect fusion gate weights and the methylation input that drove them.

        Run with masking off, so the gates reflect the model's routing of real
        measurements rather than its response to hidden ones. Used by the
        silencing probe, which asks whether a gene's expression gate falls as
        its methylation rises.

        Returns:
            Tuple of gate weights, shape ``(n_patients, n_genes, 3)``, and the
            methylation values, shape ``(n_patients, n_genes)``.
        """
        self.model.eval()
        gates, methylation = [], []
        for batch in self.val_loader:
            rna, cnv, methy = self._to_device(batch)
            out = self._forward(rna, cnv, methy, mask=False)
            gates.append(out["gate"].cpu())
            methylation.append(methy.cpu())
        return torch.cat(gates, 0), torch.cat(methylation, 0)
