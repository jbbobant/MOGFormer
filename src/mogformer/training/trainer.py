"""Training loop for the masked multi-modal reconstruction objective.

Pretraining is monitored on more than its loss. A reconstruction loss can fall
while the representation quietly collapses onto a single direction, so the
participation ratio is tracked every epoch alongside it, and per-modality losses
are reported against a predict-the-mean baseline — a head that beats the loss
but not that baseline has learned the marginal distribution, not the biology.

The structural-bias gates get their own optimiser group. They are a handful of
scalars competing with millions of weights, and at the shared learning rate they
barely move within the epoch budget; a larger rate and no weight decay lets
them find their scale. Too large a rate turns them into switches instead: at a
thousand times the base rate, a third to a half of the heads ended with their
gate at zero. Ten times is the default.

Batches carry the three modalities and, optionally, an ``observed`` mask (False
where a value was imputed, so it is never scored) and a fixed reconstruction
mask. A fixed mask on the monitoring split makes the stopping criterion a
deterministic function of the weights rather than of the epoch's mask draw.

Two embedding regularisers can join the training loss, never the monitoring
loss: a uniformity term on the normalised summary, and VICReg's variance and
covariance terms on an expanded summary. Both read a memory bank of recent
summaries (see :mod:`mogformer.training.regularisers`).

With a contrastive weight, each training batch is encoded twice. Each view
hides a different whole omics per patient, on top of its own gene masks; both
views are scored for reconstruction (their mean), and an InfoNCE loss on a
projection of the two embeddings asks them to identify the same patient. The
embedding is then rewarded for what is visible across omics.

The kept checkpoint is the one with the lowest monitoring loss by default.
With ``selection="loss_then_rank"`` it is instead, among the epochs whose
monitoring loss is within ``selection_tolerance`` of the best, the one whose
monitoring embedding has the highest RankMe: reconstruction plateaus before
the representation stops spreading, and RankMe is a label-free signal of it.
"""

from __future__ import annotations

import copy
import logging
import math
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
from mogformer.training.losses import (
    embedding_correlation_penalty,
    info_nce,
    masked_huber_dual,
    participation_ratio,
    rank_me,
    uniformity_loss,
    vicreg_covariance,
    vicreg_variance,
)
from mogformer.training.regularisers import EmbeddingMemoryBank, ProjectionHead

logger = logging.getLogger(__name__)

#: Learning-rate multiplier for the structural-bias gates. See the module note.
GATE_LR_MULTIPLIER = 10.0

#: The multiplier the archived encoders were trained with, kept so that runs
#: reproducing them (the simulation benchmark) still can.
ARCHIVED_GATE_LR_MULTIPLIER = 1000.0

#: Learning-rate schedules accepted by :class:`MaskedReconstructionTrainer`.
LR_SCHEDULES: tuple[str, ...] = ("constant", "cosine")

#: Checkpoint selection rules accepted by :class:`MaskedReconstructionTrainer`.
SELECTION_RULES: tuple[str, ...] = ("loss", "loss_then_rank")

#: Weight of VICReg's covariance term relative to its variance term, the
#: original paper's ratio (1 to 25).
VICREG_COVARIANCE_RATIO = 0.04

#: Most checkpoints kept as candidates under ``loss_then_rank``.
MAX_CANDIDATES = 32


@dataclass
class ValidationScores:
    """One epoch's monitoring diagnostics.

    Attributes:
        val: Combined monitoring loss.
        loss_global: Global-head component.
        loss_local: Local-head component.
        participation_ratio: Effective embedding dimensionality.
        rank_me: Entropy-based effective rank of the embedding.
        per_modality: Local-head loss per modality, in modality order.
        baseline: Predict-the-mean loss per modality.
    """

    val: float
    loss_global: float
    loss_local: float
    participation_ratio: float
    rank_me: float
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
        rank_me: Entropy-based effective rank of the embedding.
        contrastive: Mean training InfoNCE loss per epoch (empty without the
            contrastive objective).
        per_modality: Local-head loss per modality, in modality order.
        baseline: Predict-the-mean loss per modality, for comparison.
        grn_activation: Summed magnitude of the activating regulatory bias.
        grn_repression: Summed magnitude of the repressing regulatory bias.
        learning_rate: Base learning rate in force during each epoch.
    """

    train: list[float] = field(default_factory=list)
    val: list[float] = field(default_factory=list)
    loss_global: list[float] = field(default_factory=list)
    loss_local: list[float] = field(default_factory=list)
    participation_ratio: list[float] = field(default_factory=list)
    rank_me: list[float] = field(default_factory=list)
    contrastive: list[float] = field(default_factory=list)
    per_modality: list[list[float]] = field(default_factory=list)
    baseline: list[list[float]] = field(default_factory=list)
    grn_activation: list[float] = field(default_factory=list)
    grn_repression: list[float] = field(default_factory=list)
    learning_rate: list[float] = field(default_factory=list)


def _dominates(
    other: tuple[float, float, int, dict[str, torch.Tensor]],
    candidate: tuple[float, float, int, dict[str, torch.Tensor]],
) -> bool:
    """Whether ``other`` is at least as good on loss and RankMe, and better on one."""
    return (
        other[0] <= candidate[0]
        and other[1] >= candidate[1]
        and (other[0] < candidate[0] or other[1] > candidate[1])
    )


class MaskedReconstructionTrainer:
    """Train the self-supervised encoder and monitor it for collapse.

    Attributes:
        model: The encoder being trained.
        history: Per-epoch diagnostics.
        best_val: Best monitoring loss reached.
        selected_epoch: 1-based epoch of the checkpoint kept, 0 before
            training.
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
        gate_lr_multiplier: float = GATE_LR_MULTIPLIER,
        per_modality_loss: bool = False,
        warmup_epochs: int = 0,
        lr_schedule: str = "constant",
        min_lr: float = 0.0,
        decorrelation_penalty: float = 0.0,
        uniformity_weight: float = 0.0,
        vicreg_weight: float = 0.0,
        memory_bank_size: int = 512,
        expander_width: int = 512,
        selection: str = "loss",
        selection_tolerance: float = 0.01,
        contrastive_weight: float = 0.0,
        contrastive_temperature: float = 0.3,
        projection_width: int = 128,
        two_views: bool = False,
        view_omics_prob: float = 1.0,
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
            gate_lr_multiplier: Learning rate of the bias gates relative to
                ``lr``.
            per_modality_loss: Average the loss within each modality before
                summing, rather than over all masked entries.
            warmup_epochs: Epochs over which the learning rate rises linearly
                to ``lr``.
            lr_schedule: One of :data:`LR_SCHEDULES`. ``"cosine"`` decays from
                ``lr`` to ``min_lr`` over the epochs after warm-up.
            min_lr: Floor of the cosine schedule.
            decorrelation_penalty: Weight of the embedding decorrelation penalty
                added to the training loss (not to the monitoring loss, so the
                stopping criterion stays pure reconstruction).
            uniformity_weight: Weight of the uniformity term on the summary
                (training loss only; 0 disables it).
            vicreg_weight: Weight of VICReg's variance-plus-covariance term on
                the expanded summary (training loss only; 0 disables it).
            memory_bank_size: Recent summaries kept for both regularisers.
            expander_width: Width of the VICReg expander.
            selection: One of :data:`SELECTION_RULES`.
            selection_tolerance: Relative distance from the best monitoring
                loss within which ``loss_then_rank`` may pick a checkpoint.
            contrastive_weight: Weight of the cross-omics InfoNCE loss; above
                0 every batch is encoded as two views (training only).
            contrastive_temperature: InfoNCE temperature.
            projection_width: Output width of the contrastive head.
            two_views: Encode every batch as the two cross-omics views even
                without the contrastive loss: the control that separates the
                view regime from the loss.
            view_omics_prob: Probability, per patient and view, that the view
                hides a whole omics; when both views hide one, they hide
                different omics. 1.0 hides one in every view.

        Raises:
            ValueError: If ``lr_schedule`` or ``selection`` is unknown.
        """
        if lr_schedule not in LR_SCHEDULES:
            raise ValueError(
                f"unknown lr_schedule {lr_schedule!r}; available: {list(LR_SCHEDULES)}"
            )
        if selection not in SELECTION_RULES:
            raise ValueError(
                f"unknown selection {selection!r}; available: {list(SELECTION_RULES)}"
            )
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
        self.per_modality_loss = per_modality_loss
        self.lr = lr
        self.warmup_epochs = warmup_epochs
        self.lr_schedule = lr_schedule
        self.min_lr = min_lr
        self.decorrelation_penalty = decorrelation_penalty
        self.uniformity_weight = uniformity_weight
        self.vicreg_weight = vicreg_weight
        self.selection = selection
        self.selection_tolerance = selection_tolerance
        self.uniformity_bank = EmbeddingMemoryBank(memory_bank_size)
        self.expanded_bank = EmbeddingMemoryBank(memory_bank_size)
        self.expander: ProjectionHead | None = None
        if vicreg_weight > 0:
            self.expander = ProjectionHead(
                cast(int, model.embedding_width), expander_width
            ).to(device)
        self.contrastive_weight = contrastive_weight
        self.contrastive_temperature = contrastive_temperature
        self.contrastive_bank = EmbeddingMemoryBank(memory_bank_size)
        self.two_views = two_views or contrastive_weight > 0
        self.view_omics_prob = view_omics_prob
        self.projector: ProjectionHead | None = None
        if contrastive_weight > 0:
            self.projector = ProjectionHead(
                cast(int, model.embedding_width), 2 * projection_width, projection_width
            ).to(device)

        gate_params = [
            p for name, p in model.named_parameters() if "lambda_raw" in name
        ]
        other_params = [
            p for name, p in model.named_parameters() if "lambda_raw" not in name
        ]
        for head in self._heads():
            other_params += list(head.parameters())
        self.optimizer = torch.optim.AdamW(
            [
                {"params": other_params, "weight_decay": weight_decay},
                {
                    "params": gate_params,
                    "weight_decay": 0.0,
                    "lr": lr * gate_lr_multiplier,
                },
            ],
            lr=lr,
        )

        self.best_val = float("inf")
        self.best_state: dict[str, torch.Tensor] | None = None
        self.selected_epoch = 0
        self._candidates: list[tuple[float, float, int, dict[str, torch.Tensor]]] = []
        self.history = TrainingHistory()

    def _to_device(
        self, batch: Iterable[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Move one batch's three modality tensors onto the training device."""
        rna, cnv, methy = tuple(batch)[:3]
        return rna.to(self.device), cnv.to(self.device), methy.to(self.device)

    def _extras(
        self, batch: Iterable[torch.Tensor]
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return a batch's observed mask and fixed reconstruction mask, if any.

        Returns:
            Tuple of the observed mask and the fixed mask, each of shape
            ``(batch, n_genes, 3)`` on the training device, or None.
        """
        tensors = tuple(batch)
        observed = tensors[3].to(self.device) if len(tensors) > 3 else None
        fixed = tensors[4].to(self.device) if len(tensors) > 4 else None
        return observed, fixed

    def _forward(
        self,
        rna: torch.Tensor,
        cnv: torch.Tensor,
        methy: torch.Tensor,
        mask: bool,
        mask_bool: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run the encoder with this trainer's graph tensors."""
        return self.model(
            rna,
            cnv,
            methy,
            self.graph_pe,
            self.spd,
            self.grn,
            mask=mask,
            mask_bool=mask_bool,
        )

    def _loss(
        self, out: dict[str, torch.Tensor], observed: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Score a masked forward pass on the hidden entries that were measured."""
        scored = out["mask_bool"] if observed is None else out["mask_bool"] & observed
        out["scored"] = scored
        return masked_huber_dual(
            out["xhat_g"],
            out["xhat_l"],
            out["targets"],
            scored,
            self.lambda_global,
            self.lambda_local,
            self.delta,
            per_modality=self.per_modality_loss,
        )

    def _learning_rate_factor(self, epoch: int, max_epochs: int) -> float:
        """Return the multiple of the base rate used during a 0-based epoch."""
        if epoch < self.warmup_epochs:
            return (epoch + 1) / self.warmup_epochs
        if self.lr_schedule == "constant":
            return 1.0
        span = max(max_epochs - self.warmup_epochs, 1)
        progress = min((epoch - self.warmup_epochs) / span, 1.0)
        floor = self.min_lr / self.lr if self.lr > 0 else 0.0
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    def _regularisation(self, summary: torch.Tensor) -> torch.Tensor:
        """Return the weighted embedding regularisers for one training batch.

        Args:
            summary: The batch's patient summaries, shape ``(batch, d)``.

        Returns:
            The weighted sum of the enabled terms, a scalar tensor.
        """
        total = summary.new_zeros(())
        if self.decorrelation_penalty > 0:
            total = total + self.decorrelation_penalty * embedding_correlation_penalty(
                summary
            )
        if self.uniformity_weight > 0:
            z = F.normalize(summary, dim=1)
            total = total + self.uniformity_weight * uniformity_loss(
                z, self.uniformity_bank.values
            )
            self.uniformity_bank.push(z)
        if self.expander is not None:
            expanded = self.expander(summary)
            bank = self.expanded_bank.values
            rows = expanded if bank is None else torch.cat([expanded, bank], 0)
            total = total + self.vicreg_weight * (
                vicreg_variance(rows)
                + VICREG_COVARIANCE_RATIO * vicreg_covariance(rows)
            )
            self.expanded_bank.push(expanded)
        return total

    def _heads(self) -> list[nn.Module]:
        """Return the training-only heads that are enabled."""
        return [h for h in (self.expander, self.projector) if h is not None]

    def _two_view_loss(
        self,
        rna: torch.Tensor,
        cnv: torch.Tensor,
        methy: torch.Tensor,
        observed: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reconstruction over two cross-omics views, plus InfoNCE between them.

        Returns:
            Tuple of the mean reconstruction loss of the two views, the
            InfoNCE loss (symmetric, already unweighted), and the first view's
            embedding (for the other regularisers).
        """
        values = torch.stack([rna, cnv, methy], dim=-1)
        batch = values.shape[0]
        first = torch.randint(0, 3, (batch,), device=values.device)
        second = (first + torch.randint(1, 3, (batch,), device=values.device)) % 3
        for omics in (first, second):
            keep = torch.rand(batch, device=values.device) >= self.view_omics_prob
            omics[keep] = -1
        masker = cast(nn.Module, self.model.masker)
        reconstruction = values.new_zeros(())
        embeddings = []
        for omics in (first, second):
            mask_bool, _ = masker(values, hidden_omics=omics)
            out = self._forward(rna, cnv, methy, mask=True, mask_bool=mask_bool)
            loss, _, _ = self._loss(out, observed)
            reconstruction = reconstruction + loss / 2
            embeddings.append(out["c"])
        if self.projector is None:
            return reconstruction, reconstruction.new_zeros(()), embeddings[0]
        za, zb = self.projector(embeddings[0]), self.projector(embeddings[1])
        bank = self.contrastive_bank.values
        contrastive = (
            info_nce(za, zb, bank, self.contrastive_temperature)
            + info_nce(zb, za, bank, self.contrastive_temperature)
        ) / 2
        self.contrastive_bank.push(F.normalize(torch.cat([za, zb]), dim=1))
        return reconstruction, contrastive, embeddings[0]

    def _keep_candidate(self, epoch: int, scores: ValidationScores) -> None:
        """Update the checkpoints ``loss_then_rank`` may still select.

        Keeps the epochs within tolerance of the best loss that no other kept
        epoch beats on both loss and RankMe, so the list stays short. Called
        before ``best_val`` takes this epoch into account, hence the ``min``.
        """
        limit = min(self.best_val, scores.val) * (1.0 + self.selection_tolerance)
        if scores.val <= limit:
            state = {
                k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
            }
            self._candidates.append((scores.val, scores.rank_me, epoch, state))
        kept = [c for c in self._candidates if c[0] <= limit]
        kept = [c for c in kept if not any(_dominates(o, c) for o in kept)]
        kept.sort(key=lambda c: c[1], reverse=True)
        self._candidates = kept[:MAX_CANDIDATES]

    @torch.no_grad()
    def attention_gates(self) -> list[list[float]]:
        """Return each block's per-head distance-bias gate, ``softplus(λ)``.

        Returns:
            One list per block, one value per head; empty when the model has
            no gates.
        """
        transformer = cast(GlobalGraphTransformer, self.model.global_transformer)
        gates = []
        for layer in transformer.layers:
            attention = cast(StructuralAttentionBlock, layer).attn
            if attention.lambda_gate:
                gates.append(F.softplus(attention.lambda_raw).cpu().tolist())
        return gates

    def train_epoch(self) -> float:
        """Run one pass over the training batches.

        Returns:
            Mean training loss over the epoch.
        """
        self.model.train()
        for head in self._heads():
            head.train()
        total, contrastive_total, n_batches = 0.0, 0.0, 0
        for batch in self.train_loader:
            rna, cnv, methy = self._to_device(batch)
            observed, _ = self._extras(batch)
            self.optimizer.zero_grad()
            if self.two_views:
                loss, contrastive, summary = self._two_view_loss(
                    rna, cnv, methy, observed
                )
                loss = loss + self.contrastive_weight * contrastive
                contrastive_total += contrastive.item()
            else:
                out = self._forward(rna, cnv, methy, mask=True)
                loss, _, _ = self._loss(out, observed)
                summary = out["c"]
            loss = loss + self._regularisation(summary)
            loss.backward()
            parameters = list(self.model.parameters())
            for head in self._heads():
                parameters += list(head.parameters())
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
            self.optimizer.step()
            total += loss.item()
            n_batches += 1
        if self.projector is not None:
            self.history.contrastive.append(contrastive_total / max(n_batches, 1))
        return total / max(n_batches, 1)

    @torch.no_grad()
    def validate_epoch(self) -> ValidationScores:
        """Score the monitoring split and collect collapse diagnostics.

        Batches carrying a fixed mask are scored on it; others draw a fresh
        mask, which makes the score noisier from epoch to epoch.

        Returns:
            Mapping with the monitoring loss, both head components, the
            participation ratio and RankMe, and per-modality losses beside their
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
            observed, fixed = self._extras(batch)
            out = self._forward(rna, cnv, methy, mask=True, mask_bool=fixed)
            loss, loss_global, loss_local = self._loss(out, observed)
            total += loss.item()
            global_total += loss_global.item()
            local_total += loss_local.item()
            n_batches += 1
            embeddings.append(out["c"].cpu())

            mask_bool, targets, local = (
                out["scored"],
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

        summaries = torch.cat(embeddings, 0)
        return ValidationScores(
            val=total / max(n_batches, 1),
            loss_global=global_total / max(n_batches, 1),
            loss_local=local_total / max(n_batches, 1),
            participation_ratio=participation_ratio(summaries),
            rank_me=rank_me(summaries),
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

        The selected state is restored before returning, so the trainer always
        ends holding the model its selection rule chose rather than the last
        one. Stopping always follows the monitoring loss.

        Args:
            max_epochs: Upper bound on epochs.
            patience: Epochs without improvement before stopping.
            log_every: Epoch interval for progress logging.

        Returns:
            The best monitoring loss reached.
        """
        epochs_without_improvement = 0
        base_rates = [group["lr"] for group in self.optimizer.param_groups]

        for epoch in range(1, max_epochs + 1):
            factor = self._learning_rate_factor(epoch - 1, max_epochs)
            for group, base in zip(
                self.optimizer.param_groups, base_rates, strict=True
            ):
                group["lr"] = base * factor
            self.history.learning_rate.append(self.lr * factor)
            train_loss = self.train_epoch()
            scores = self.validate_epoch()

            self.history.train.append(train_loss)
            self.history.val.append(scores.val)
            self.history.loss_global.append(scores.loss_global)
            self.history.loss_local.append(scores.loss_local)
            self.history.participation_ratio.append(scores.participation_ratio)
            self.history.rank_me.append(scores.rank_me)
            self.history.per_modality.append(list(scores.per_modality))
            self.history.baseline.append(list(scores.baseline))
            self._record_gate_magnitudes()

            if epoch % log_every == 0:
                per_modality = scores.per_modality
                baseline = scores.baseline
                logger.info(
                    "epoch %03d | train %.4f | val %.4f (global %.4f local %.4f) "
                    "| participation %.2f rankme %.2f | per-modality %s vs baseline %s",
                    epoch,
                    train_loss,
                    scores.val,
                    scores.loss_global,
                    scores.loss_local,
                    scores.participation_ratio,
                    scores.rank_me,
                    "/".join(f"{v:.3f}" for v in per_modality),
                    "/".join(f"{v:.3f}" for v in baseline),
                )

            if self.selection == "loss_then_rank":
                self._keep_candidate(epoch, scores)
            if scores.val < self.best_val - 1e-5:
                self.best_val = scores.val
                self.best_state = copy.deepcopy(self.model.state_dict())
                self.selected_epoch = epoch
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

        if self.selection == "loss_then_rank" and self._candidates:
            limit = self.best_val * (1.0 + self.selection_tolerance)
            _, best_rank, epoch, state = max(
                (c for c in self._candidates if c[0] <= limit), key=lambda c: c[1]
            )
            logger.info(
                "selected epoch %d (RankMe %.2f) within %.1f%% of the best loss",
                epoch,
                best_rank,
                100 * self.selection_tolerance,
            )
            self.selected_epoch = epoch
            self.model.load_state_dict(state)
        elif self.best_state is not None:
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
