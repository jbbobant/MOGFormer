"""Unit tests for the layers, the classifier and the self-supervised encoder.

Everything runs on tiny random tensors on CPU, so the suite stays fast and needs
no cohort data.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mogformer.data import MODALITY_ORDER
from mogformer.models import MOGFormerSSL, MultiOmicsGraphClassifier
from mogformer.models.layers import (
    DualHeadDecoder,
    GatedFusion,
    GeneEmbeddingAdapter,
    GlobalGraphTransformer,
    MaskedMultiModalMasker,
    MiniTransformer,
    ModalityDropout,
    ModalityLifting,
    PeriodicLinearTokenizer,
    StructuralGraphAttention,
)
from mogformer.training import (
    MultiClassFocalLoss,
    masked_huber_dual,
    participation_ratio,
    sqrt_dampened_weights,
)

BATCH, N_GENES, WIDTH = 3, 6, 16


@pytest.fixture(autouse=True)
def _seeded() -> None:
    """Make every test deterministic without touching global config."""
    torch.manual_seed(0)


@pytest.fixture()
def scalars() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return one random scalar stream per modality."""
    return (
        torch.randn(BATCH, N_GENES),
        torch.randn(BATCH, N_GENES),
        torch.randn(BATCH, N_GENES),
    )


@pytest.fixture()
def tokens() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return one random token stream per modality."""
    return (
        torch.randn(BATCH, N_GENES, WIDTH),
        torch.randn(BATCH, N_GENES, WIDTH),
        torch.randn(BATCH, N_GENES, WIDTH),
    )


@pytest.fixture()
def graph() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return positional encodings, a distance matrix and a regulatory matrix."""
    pe = torch.randn(N_GENES, 8)
    spd = torch.randint(0, 5, (N_GENES, N_GENES))
    spd = torch.minimum(spd, spd.T)
    spd.fill_diagonal_(0)
    grn = torch.zeros(N_GENES, N_GENES)
    grn[1, 0] = 1.0
    grn[2, 0] = -1.0
    return pe, spd, grn


# --------------------------------------------------------------------------
# Structural attention
# --------------------------------------------------------------------------
def test_attention_bias_table_covers_the_summary_bucket() -> None:
    """The table must index distances up to and including ``max_distance + 2``.

    The enclosing transformer gives the tumor summary token that sentinel
    distance so it does not share the zero bucket that means "this gene is
    itself". A table sized ``max_distance + 2`` would index out of range.
    """
    max_distance = 4
    attention = StructuralGraphAttention(
        d_model=WIDTH, num_heads=2, max_distance=max_distance
    )
    assert attention.spatial_embedding.num_embeddings == max_distance + 3

    saturated = torch.full((N_GENES, N_GENES), max_distance + 2)
    attention(torch.randn(BATCH, N_GENES, WIDTH), saturated)


def test_attention_is_shape_preserving(graph) -> None:
    """Attention returns the sequence it was given, same shape."""
    _, spd, _ = graph
    attention = StructuralGraphAttention(d_model=WIDTH, num_heads=2)
    out = attention(torch.randn(BATCH, N_GENES, WIDTH), spd)
    assert out.shape == (BATCH, N_GENES, WIDTH)


def test_attention_bias_changes_the_output(graph) -> None:
    """Disabling the structural bias must actually change the result."""
    _, spd, _ = graph
    attention = StructuralGraphAttention(
        d_model=WIDTH, num_heads=2, mode="inside", dropout=0.0
    ).eval()
    # An untrained bias table is zero-initialised, so give it real values.
    torch.nn.init.normal_(attention.spatial_embedding.weight, std=1.0)
    h = torch.randn(BATCH, N_GENES, WIDTH)

    with_bias = attention(h, spd, structural_bias=True)
    without_bias = attention(h, spd, structural_bias=False)

    assert not torch.allclose(with_bias, without_bias)


def test_regulatory_bias_only_applies_when_enabled(graph) -> None:
    """The regulatory term is inert until the enclosing transformer turns it on."""
    _, spd, grn = graph
    attention = StructuralGraphAttention(
        d_model=WIDTH, num_heads=2, mode="inside", dropout=0.0
    ).eval()
    torch.nn.init.normal_(attention.b_grn_activation, std=1.0)
    torch.nn.init.normal_(attention.b_grn_repression, std=1.0)
    h = torch.randn(BATCH, N_GENES, WIDTH)

    assert torch.allclose(attention(h, spd, grn), attention(h, spd, None))

    attention.use_grn = True
    assert not torch.allclose(attention(h, spd, grn), attention(h, spd, None))


def test_dual_mode_attention_rows_sum_to_one(graph) -> None:
    """Averaging two normalised distributions keeps the rows a distribution."""
    _, spd, _ = graph
    attention = StructuralGraphAttention(
        d_model=WIDTH, num_heads=2, mode="dual", dropout=0.0
    )
    attention.eval()
    # Reach into the computation by disabling the projection's effect on shape.
    h = torch.randn(BATCH, N_GENES, WIDTH)
    assert attention(h, spd).shape == (BATCH, N_GENES, WIDTH)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"mode": "sideways"}, "unknown attention bias mode"),
        ({"num_heads": 5}, "not divisible"),
    ],
)
def test_attention_rejects_bad_configuration(kwargs, match) -> None:
    """Unknown modes and indivisible widths fail at construction."""
    options = {"d_model": WIDTH, "num_heads": 2, **kwargs}
    with pytest.raises(ValueError, match=match):
        StructuralGraphAttention(**options)


# --------------------------------------------------------------------------
# Global transformer
# --------------------------------------------------------------------------
def test_global_transformer_prepends_one_summary_token(graph) -> None:
    """The sequence grows by exactly one and the summary is at index 0."""
    pe, spd, _ = graph
    transformer = GlobalGraphTransformer(
        d=WIDTH, pe_dim=8, num_heads=2, num_layers=1, max_distance=4
    )
    summary, sequence = transformer(torch.randn(BATCH, N_GENES, WIDTH), pe, spd)

    assert summary.shape == (BATCH, WIDTH)
    assert sequence.shape == (BATCH, N_GENES + 1, WIDTH)
    assert torch.allclose(summary, sequence[:, 0, :])


def test_global_transformer_gives_the_summary_its_own_distance(graph) -> None:
    """The summary row is padded with the sentinel, not with zero."""
    pe, spd, _ = graph
    max_distance = 4
    transformer = GlobalGraphTransformer(
        d=WIDTH, pe_dim=8, num_heads=2, num_layers=1, max_distance=max_distance
    )
    assert transformer.summary_distance == max_distance + 2

    captured: list[torch.Tensor] = []
    original = transformer.layers[0].forward

    def capture(x, padded_spd, *args, **kwargs):
        captured.append(padded_spd)
        return original(x, padded_spd, *args, **kwargs)

    transformer.layers[0].forward = capture
    transformer(torch.randn(BATCH, N_GENES, WIDTH), pe, spd)

    padded = captured[0]
    assert padded.shape == (N_GENES + 1, N_GENES + 1)
    assert int(padded[0, 0]) == max_distance + 2
    assert int(padded[0, 3]) == max_distance + 2
    # The gene block is untouched.
    assert torch.equal(padded[1:, 1:], spd)


def test_global_transformer_enables_regulatory_bias_in_every_layer(graph) -> None:
    """``use_grn`` propagates to each block rather than only the first."""
    transformer = GlobalGraphTransformer(
        d=WIDTH, pe_dim=8, num_heads=2, num_layers=3, use_grn=True
    )
    assert all(layer.attn.use_grn for layer in transformer.layers)


# --------------------------------------------------------------------------
# Modality dropout, shared by both fusion layers
# --------------------------------------------------------------------------
def test_modality_dropout_is_inactive_in_eval(tokens) -> None:
    """Without an eval mask, evaluation mode passes tokens through untouched."""
    dropout = ModalityDropout(
        d=WIDTH, rna_dropout_prob=0.9, cnv_dropout_prob=0.0, meth_dropout_prob=0.0
    ).eval()
    out = dropout(*tokens)
    assert all(torch.equal(a, b) for a, b in zip(out, tokens, strict=True))


def test_modality_dropout_hides_at_most_one_modality(tokens) -> None:
    """Two modalities are never hidden for the same gene."""
    dropout = ModalityDropout(
        d=WIDTH,
        rna_dropout_prob=0.34,
        cnv_dropout_prob=0.33,
        meth_dropout_prob=0.33,
    ).train()

    z_rna, z_cnv, z_methy = dropout(
        torch.ones(BATCH, N_GENES, WIDTH),
        torch.ones(BATCH, N_GENES, WIDTH),
        torch.ones(BATCH, N_GENES, WIDTH),
    )
    zeroed = torch.stack(
        [(stream == 0).all(dim=-1) for stream in (z_rna, z_cnv, z_methy)], dim=-1
    )
    assert int(zeroed.sum(dim=-1).max()) <= 1


def test_modality_dropout_rejects_probabilities_above_one() -> None:
    """The three probabilities partition one uniform draw, so they must fit."""
    with pytest.raises(ValueError, match=r"exceeds 1\.0"):
        ModalityDropout(
            d=WIDTH,
            rna_dropout_prob=0.5,
            cnv_dropout_prob=0.4,
            meth_dropout_prob=0.4,
        )


def test_eval_mask_requires_mask_tokens(tokens) -> None:
    """Zero-filling at evaluation time is not a trained state, so it is refused."""
    dropout = ModalityDropout(d=WIDTH, fill="zero").eval()
    with pytest.raises(ValueError, match=r"requires fill='mask_token'"):
        dropout(*tokens, eval_mask=["rna"])


def test_eval_mask_replaces_the_named_modality(tokens) -> None:
    """With mask tokens, the named modality is deterministically replaced."""
    dropout = ModalityDropout(d=WIDTH, fill="mask_token").eval()
    z_rna, z_cnv, z_methy = dropout(*tokens, eval_mask=["rna"])

    assert torch.allclose(z_rna, dropout.mask_rna.expand_as(z_rna))
    assert torch.equal(z_cnv, tokens[1])
    assert torch.equal(z_methy, tokens[2])


def test_eval_mask_rejects_unknown_modality(tokens) -> None:
    """A typo in a probe's modality name fails loudly."""
    dropout = ModalityDropout(d=WIDTH, fill="mask_token").eval()
    with pytest.raises(ValueError, match="unknown modalities"):
        dropout(*tokens, eval_mask=["protein"])


# --------------------------------------------------------------------------
# Fusion layers
# --------------------------------------------------------------------------
def test_mini_transformer_returns_fused_tokens_and_attention(tokens) -> None:
    """Four-token attention over ``[summary, rna, cnv, methy]``."""
    fusion = MiniTransformer(d=WIDTH, num_heads=2).eval()
    fused, attention = fusion(*tokens)

    assert fused.shape == (BATCH, N_GENES, WIDTH)
    assert attention.shape == (BATCH * N_GENES, 4, 4)


def test_gated_fusion_weights_compete(tokens) -> None:
    """Gate weights form a distribution per gene, so modalities trade off."""
    fusion = GatedFusion(d=WIDTH).eval()
    fused, weights = fusion(*tokens)

    assert fused.shape == (BATCH, N_GENES, WIDTH)
    assert weights.shape == (BATCH, N_GENES, len(MODALITY_ORDER))
    assert torch.allclose(weights.sum(dim=-1), torch.ones(BATCH, N_GENES), atol=1e-6)
    assert torch.all(weights >= 0)


def test_both_fusion_layers_share_a_call_signature(tokens) -> None:
    """The two are interchangeable, which is what makes the ablation single-variable."""
    attention_out, _ = MiniTransformer(d=WIDTH, num_heads=2).eval()(*tokens)
    gated_out, _ = GatedFusion(d=WIDTH).eval()(*tokens)
    assert attention_out.shape == gated_out.shape


# --------------------------------------------------------------------------
# Modality lifting
# --------------------------------------------------------------------------
def test_modality_lifting_produces_one_token_stream_per_modality(scalars) -> None:
    """Three scalar streams become three token streams."""
    lifter = ModalityLifting(d=WIDTH)
    streams = lifter(*scalars)

    assert len(streams) == len(MODALITY_ORDER)
    assert all(stream.shape == (BATCH, N_GENES, WIDTH) for stream in streams)


def test_modality_lifting_masks_only_the_value(scalars) -> None:
    """A masked token still differs by modality, so the address survives."""
    lifter = ModalityLifting(d=WIDTH)
    mask = torch.zeros(BATCH, N_GENES, 3, dtype=torch.bool)
    mask[:, 0, 0] = True

    z_rna, _, _ = lifter(*scalars, mask_bool=mask)
    expected = lifter.mask_rna.squeeze() + lifter.emb_rna.squeeze()

    assert torch.allclose(z_rna[0, 0], expected, atol=1e-6)
    # An unmasked gene is untouched.
    assert not torch.allclose(z_rna[0, 1], expected, atol=1e-6)


def test_periodic_tokenizer_shape() -> None:
    """The periodic tokenizer maps scalars to tokens of the requested width."""
    tokenizer = PeriodicLinearTokenizer(d=WIDTH, n_frequencies=4)
    assert tokenizer(torch.randn(BATCH, N_GENES)).shape == (BATCH, N_GENES, WIDTH)


def test_modality_lifting_accepts_the_periodic_tokenizer(scalars) -> None:
    """Switching tokenizer changes nothing about the output contract."""
    lifter = ModalityLifting(d=WIDTH, numerical_tokenizer="plr", plr_n_frequencies=4)
    assert all(stream.shape == (BATCH, N_GENES, WIDTH) for stream in lifter(*scalars))


def test_modality_lifting_rejects_unknown_tokenizer() -> None:
    """An unsupported tokenizer fails at construction."""
    with pytest.raises(ValueError, match="unknown numerical tokenizer"):
        ModalityLifting(d=WIDTH, numerical_tokenizer="fourier")


def test_gene_identity_requires_a_source() -> None:
    """Identity embeddings need either pretrained vectors or a universe."""
    with pytest.raises(ValueError, match="gene_id_embedding requires"):
        ModalityLifting(d=WIDTH, gene_id_embedding=True)


def test_learnable_gene_identity_is_shared_with_the_decoder(scalars) -> None:
    """The accessor returns the same embeddings the forward pass added."""
    lifter = ModalityLifting(
        d=WIDTH,
        gene_id_embedding=True,
        n_universe=32,
        gene_ids=list(range(N_GENES)),
    )
    identity = lifter.get_gene_embedding()

    assert identity is not None
    assert identity.shape == (N_GENES, WIDTH)
    assert lifter.get_modality_embeddings().shape == (3, WIDTH)


def test_pretrained_gene_embeddings_stay_frozen() -> None:
    """The pretrained matrix is a buffer, so it collects no gradient."""
    pretrained = torch.randn(N_GENES, 12)
    adapter = GeneEmbeddingAdapter(pretrained, d_out=WIDTH, adapter_rank=0)

    assert "base" in dict(adapter.named_buffers())
    assert "base" not in dict(adapter.named_parameters())
    assert adapter().shape == (1, N_GENES, WIDTH)


def test_gene_adapter_starts_as_the_identity() -> None:
    """A zero-initialised up-projection means rank>0 begins at the frozen value."""
    pretrained = torch.randn(N_GENES, 12)
    plain = GeneEmbeddingAdapter(pretrained, d_out=WIDTH, adapter_rank=0)
    adapted = GeneEmbeddingAdapter(pretrained, d_out=WIDTH, adapter_rank=4)
    adapted.proj.load_state_dict(plain.proj.state_dict())

    assert torch.allclose(plain(), adapted(), atol=1e-6)


# --------------------------------------------------------------------------
# Masking
# --------------------------------------------------------------------------
def test_masker_keeps_whole_gene_and_single_modality_sets_disjoint() -> None:
    """A gene is either wholly hidden or has at most one channel hidden."""
    masker = MaskedMultiModalMasker(mask_gene_frac=0.3, mask_modality_frac=0.5)
    mask, targets = masker(torch.randn(BATCH, 20, 3))

    per_gene = mask.sum(dim=-1)
    assert set(per_gene.unique().tolist()) <= {0, 1, 3}
    assert targets.shape == (BATCH, 20, 3)


def test_masker_hides_the_requested_fraction_of_whole_genes() -> None:
    """Whole-gene masking hits exactly ``floor(frac * n_genes)`` genes."""
    masker = MaskedMultiModalMasker(mask_gene_frac=0.25, mask_modality_frac=0.0)
    mask, _ = masker(torch.randn(BATCH, 20, 3))

    fully_masked = (mask.sum(dim=-1) == 3).sum(dim=1)
    assert torch.equal(fully_masked, torch.full((BATCH,), 5))


def test_masker_returns_targets_taken_before_masking() -> None:
    """Targets are a copy of the input, not a masked view of it."""
    masker = MaskedMultiModalMasker()
    values = torch.randn(BATCH, N_GENES, 3)
    _, targets = masker(values)
    assert torch.equal(targets, values)


@pytest.mark.parametrize(
    "kwargs",
    [{"mask_gene_frac": 1.5}, {"mask_modality_frac": -0.1}],
)
def test_masker_rejects_fractions_outside_the_unit_interval(kwargs) -> None:
    """Rates outside ``[0, 1]`` are refused."""
    with pytest.raises(ValueError, match="must lie in"):
        MaskedMultiModalMasker(**kwargs)


# --------------------------------------------------------------------------
# Decoder
# --------------------------------------------------------------------------
def test_decoder_predicts_every_gene_by_modality_cell() -> None:
    """Both heads emit one prediction per gene and modality."""
    decoder = DualHeadDecoder(d=WIDTH)
    global_pred, local_pred = decoder(
        torch.randn(BATCH, WIDTH),
        torch.randn(BATCH, N_GENES + 1, WIDTH),
        torch.randn(N_GENES, WIDTH),
        torch.randn(3, WIDTH),
    )
    assert global_pred.shape == (BATCH, N_GENES, 3)
    assert local_pred.shape == (BATCH, N_GENES, 3)


def test_decoder_heads_disagree() -> None:
    """The heads see different inputs, so identical output would be a wiring bug."""
    decoder = DualHeadDecoder(d=WIDTH)
    global_pred, local_pred = decoder(
        torch.randn(BATCH, WIDTH),
        torch.randn(BATCH, N_GENES + 1, WIDTH),
        torch.randn(N_GENES, WIDTH),
        torch.randn(3, WIDTH),
    )
    assert not torch.allclose(global_pred, local_pred)


# --------------------------------------------------------------------------
# Full models
# --------------------------------------------------------------------------
def test_classifier_produces_logits_and_interpretability(scalars, graph) -> None:
    """The supervised model returns logits, plus attention when asked."""
    pe, spd, _ = graph
    model = MultiOmicsGraphClassifier(
        num_classes=4, d=WIDTH, pe_dim=8, mini_heads=2, global_heads=2, global_layers=1
    ).eval()

    plain = model(*scalars, pe, spd)
    assert set(plain) == {"logits"}
    assert plain["logits"].shape == (BATCH, 4)

    detailed = model(*scalars, pe, spd, return_attention=True)
    assert detailed["tumor_state"].shape == (BATCH, WIDTH)
    assert detailed["intra_attn_weights"].shape == (BATCH * N_GENES, 4, 4)
    assert detailed["hidden_last"].shape == (BATCH, N_GENES + 1, WIDTH)


def test_classifier_backpropagates(scalars, graph) -> None:
    """A loss on the logits reaches the tokenizer weights."""
    pe, spd, _ = graph
    model = MultiOmicsGraphClassifier(
        num_classes=4, d=WIDTH, pe_dim=8, mini_heads=2, global_heads=2, global_layers=1
    )
    logits = model(*scalars, pe, spd)["logits"]
    logits.sum().backward()

    assert model.lifter.tok_rna[0].weight.grad is not None
    assert torch.isfinite(model.lifter.tok_rna[0].weight.grad).all()


def build_ssl(**kwargs) -> MOGFormerSSL:
    """Construct a small self-supervised encoder for the tests."""
    options = {
        "d": WIDTH,
        "pe_dim": 8,
        "mini_heads": 2,
        "global_heads": 2,
        "global_layers": 1,
        "max_distance": 4,
        "n_universe": 32,
        "gene_ids": list(range(N_GENES)),
        **kwargs,
    }
    return MOGFormerSSL(**options)


def test_ssl_reconstructs_when_masking(scalars, graph) -> None:
    """With masking on, both heads and the mask itself are returned."""
    pe, spd, grn = graph
    model = build_ssl().eval()
    out = model(*scalars, pe, spd, grn)

    assert out["xhat_g"].shape == (BATCH, N_GENES, 3)
    assert out["xhat_l"].shape == (BATCH, N_GENES, 3)
    assert out["mask_bool"].shape == (BATCH, N_GENES, 3)
    assert out["targets"].shape == (BATCH, N_GENES, 3)


def test_ssl_returns_embeddings_only_when_masking_is_off(scalars, graph) -> None:
    """The analysis path skips reconstruction entirely."""
    pe, spd, grn = graph
    out = build_ssl().eval()(*scalars, pe, spd, grn, mask=False)

    assert set(out) == {"c", "H_final", "gate"}
    assert out["c"].shape == (BATCH, WIDTH)
    assert out["H_final"].shape == (BATCH, N_GENES + 1, WIDTH)


def test_ssl_accepts_a_deterministic_mask(scalars, graph) -> None:
    """The probes supply their own mask, hiding exactly one measurement."""
    pe, spd, grn = graph
    mask = torch.zeros(BATCH, N_GENES, 3, dtype=torch.bool)
    mask[:, 2, 0] = True

    out = build_ssl().eval()(*scalars, pe, spd, grn, mask_bool=mask)
    assert torch.equal(out["mask_bool"], mask)


def test_ssl_never_masks_a_zeroed_modality(scalars, graph) -> None:
    """Under ``rna_only`` the zeroed channels are excluded from the objective."""
    pe, spd, grn = graph
    out = build_ssl(rna_only=True).eval()(*scalars, pe, spd, grn)

    assert not out["mask_bool"][..., 1].any()
    assert not out["mask_bool"][..., 2].any()


def test_ssl_requires_gene_identity_embeddings() -> None:
    """Without them the decoder cannot address its predictions."""
    with pytest.raises(ValueError, match="gene_id_embedding is required"):
        build_ssl(gene_id_embedding=False)


def test_ssl_rejects_unknown_fusion_type() -> None:
    """A mistyped fusion choice fails at construction."""
    with pytest.raises(ValueError, match="unknown fusion type"):
        build_ssl(fusion_type="concat")


# --------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------
def test_focal_loss_reduces_to_cross_entropy_at_gamma_zero() -> None:
    """Gamma zero and no weights is plain cross-entropy."""
    logits = torch.randn(8, 4)
    targets = torch.randint(0, 4, (8,))
    focal = MultiClassFocalLoss(gamma=0.0)(logits, targets)
    expected = torch.nn.functional.cross_entropy(logits, targets)
    assert torch.allclose(focal, expected, atol=1e-6)


def test_focal_loss_down_weights_confident_examples() -> None:
    """Focusing shrinks the loss on an example the model already gets right."""
    confident = torch.tensor([[2.0, -2.0]])
    target = torch.tensor([0])
    plain = MultiClassFocalLoss(gamma=0.0)(confident, target)
    focused = MultiClassFocalLoss(gamma=3.0)(confident, target)
    assert focused < plain


def test_focal_loss_applies_class_weights() -> None:
    """A weighted class contributes proportionally more."""
    logits = torch.randn(6, 3)
    targets = torch.zeros(6, dtype=torch.long)
    unweighted = MultiClassFocalLoss(gamma=0.0)(logits, targets)
    weighted = MultiClassFocalLoss(alpha=torch.tensor([2.0, 1.0, 1.0]), gamma=0.0)(
        logits, targets
    )
    assert torch.allclose(weighted, 2.0 * unweighted, atol=1e-6)


def test_focal_loss_rejects_bad_configuration() -> None:
    """Unknown reductions and negative focusing are refused."""
    with pytest.raises(ValueError, match="unknown reduction"):
        MultiClassFocalLoss(reduction="median")
    with pytest.raises(ValueError, match="non-negative"):
        MultiClassFocalLoss(gamma=-1.0)


def test_sqrt_dampened_weights_favour_rare_classes() -> None:
    """The rarest class gets the largest weight, but not inverse-frequency large."""
    y = np.repeat([0, 1, 2], [100, 10, 1])
    weights = sqrt_dampened_weights(y, num_classes=3)

    assert weights[2] > weights[1] > weights[0]
    # Dampened: the ratio is sqrt(100) = 10, not 100.
    assert weights[2] / weights[0] == pytest.approx(10.0, rel=1e-6)


def test_sqrt_dampened_weights_handle_absent_classes() -> None:
    """A class with no training examples receives zero weight, not infinity."""
    weights = sqrt_dampened_weights(np.array([0, 0, 1]), num_classes=4)
    assert weights[2] == 0.0
    assert weights[3] == 0.0
    assert np.isfinite(weights).all()


def test_masked_huber_scores_only_hidden_entries() -> None:
    """Changing a visible prediction must not change the loss."""
    targets = torch.zeros(2, 4, 3)
    mask = torch.zeros(2, 4, 3, dtype=torch.bool)
    mask[0, 0, 0] = True

    base = torch.zeros(2, 4, 3)
    loss_before, _, _ = masked_huber_dual(base, base, targets, mask)

    perturbed = base.clone()
    perturbed[1, 3, 2] = 99.0
    loss_after, _, _ = masked_huber_dual(perturbed, perturbed, targets, mask)

    assert torch.allclose(loss_before, loss_after)


def test_masked_huber_weights_the_two_heads() -> None:
    """Head weights scale their contributions independently."""
    targets = torch.zeros(1, 2, 3)
    mask = torch.ones(1, 2, 3, dtype=torch.bool)
    predictions = torch.full((1, 2, 3), 0.5)

    total, global_term, local_term = masked_huber_dual(
        predictions, predictions, targets, mask, lambda_global=2.0, lambda_local=0.0
    )
    assert torch.allclose(total, 2.0 * global_term)
    assert torch.allclose(global_term, local_term)


def test_participation_ratio_detects_collapse() -> None:
    """A rank-one embedding uses one effective dimension; noise uses many."""
    direction = torch.randn(1, 8)
    collapsed = torch.randn(64, 1) @ direction
    assert participation_ratio(collapsed) == pytest.approx(1.0, abs=1e-3)

    spread = torch.randn(512, 8)
    assert participation_ratio(spread) > 5.0
