"""Tests for the self-supervised encoder's training path.

Each test pins one property the encoder's correctness depends on: fused and
explicit attention agree, imputed values are never scored, each omics weighs
the same in the loss, the stopping criterion is deterministic, the patient
splits are disjoint and shared across seeds, and a run writes everything a
downstream analysis reads. Everything is synthetic and runs on the CPU.
"""

from __future__ import annotations

import json
import math
from itertools import pairwise

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from mogformer.analysis.embedding import (
    READOUTS,
    PatientSplits,
    heldout_reconstruction,
    run_pretraining,
    split_pretraining_patients,
)
from mogformer.config import ExperimentConfig
from mogformer.data.omics import OmicsData
from mogformer.data.preprocess import MultiOmicsTransformer
from mogformer.models import MOGFormerSSL
from mogformer.models.layers import (
    DualHeadDecoder,
    GlobalGraphTransformer,
    MaskedMultiModalMasker,
    StructuralGraphAttention,
)
from mogformer.training import masked_huber_dual
from mogformer.training.trainer import MaskedReconstructionTrainer

N_GENES, WIDTH = 6, 16


@pytest.fixture(autouse=True)
def _seeded() -> None:
    """Make every test deterministic."""
    torch.manual_seed(0)


def _spd(isolated: int | None = None) -> torch.Tensor:
    """Return a path-graph distance matrix, optionally with one isolated gene."""
    index = torch.arange(N_GENES)
    spd = (index[:, None] - index[None, :]).abs().clamp(max=5)
    if isolated is not None:
        spd[isolated, :] = 5
        spd[:, isolated] = 5
        spd[isolated, isolated] = 0
    return spd


def _encoder(**kwargs) -> MOGFormerSSL:
    """Build a small encoder whose identity table has one row per gene."""
    options = {
        "d": WIDTH,
        "pe_dim": 4,
        "mini_heads": 2,
        "global_heads": 2,
        "global_layers": 1,
        "max_distance": 4,
        "n_universe": N_GENES,
        "gene_ids": list(range(N_GENES)),
        "lambda_gate": True,
        **kwargs,
    }
    return MOGFormerSSL(**options)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
@pytest.mark.parametrize("lambda_gate", [False, True])
def test_fused_attention_equals_the_explicit_computation(lambda_gate) -> None:
    """The fused kernel and the explicit softmax give the same output."""
    attention = StructuralGraphAttention(
        WIDTH, 2, max_distance=4, lambda_gate=lambda_gate
    )
    attention.use_grn = True
    attention.eval()
    with torch.no_grad():
        attention.spatial_embedding.weight.normal_()
        attention.b_grn_activation.fill_(0.7)
    tokens = torch.randn(3, N_GENES, WIDTH)
    grn = torch.zeros(N_GENES, N_GENES)
    grn[1, 0] = 1.0

    fused = attention(tokens, _spd(), grn)
    weights = attention.attention_weights(tokens, _spd(), grn)
    _, _, value = attention._project(tokens)
    explicit = attention.out_proj(
        torch.matmul(weights, value).transpose(1, 2).reshape(3, N_GENES, WIDTH)
    )

    torch.testing.assert_close(fused, explicit, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(weights.sum(-1), torch.ones(3, 2, N_GENES))


def test_fused_attention_propagates_gradient_to_the_bias() -> None:
    """The distance-bias table and the gates still learn through the kernel."""
    attention = StructuralGraphAttention(WIDTH, 2, max_distance=4, lambda_gate=True)
    attention(torch.randn(2, N_GENES, WIDTH), _spd()).sum().backward()

    assert attention.spatial_embedding.weight.grad.abs().sum() > 0
    assert attention.lambda_raw.grad.abs().sum() > 0


def test_only_isolated_genes_receive_the_isolated_code() -> None:
    """A gene with no edge gets the learned vector; connected genes do not."""
    transformer = GlobalGraphTransformer(
        d=WIDTH,
        pe_dim=4,
        num_heads=2,
        num_layers=1,
        max_distance=4,
        isolated_token=True,
    )
    plain = GlobalGraphTransformer(
        d=WIDTH, pe_dim=4, num_heads=2, num_layers=1, max_distance=4
    )
    plain.load_state_dict(transformer.state_dict(), strict=False)
    with torch.no_grad():
        transformer.isolated_embedding.fill_(10.0)
    captured: dict[str, torch.Tensor] = {}

    def recorder(key: str):
        def hook(module, args) -> None:
            captured[key] = args[0].clone()

        return hook

    transformer.layers[0].register_forward_pre_hook(recorder("with"))
    plain.layers[0].register_forward_pre_hook(recorder("without"))
    tokens, pe = torch.randn(1, N_GENES, WIDTH), torch.randn(N_GENES, 4)
    transformer.eval()(tokens, pe, _spd(isolated=2))
    plain.eval()(tokens, pe, _spd(isolated=2))

    shift = (captured["with"] - captured["without"])[0, 1:].abs().sum(-1)
    assert shift[2] > 0
    assert torch.all(shift[[0, 1, 3, 4, 5]] == 0)


def test_the_decoder_offset_can_be_removed_and_is_kept_by_default() -> None:
    """Without the offset there is no beta network; archived weights still load."""
    assert hasattr(DualHeadDecoder(WIDTH), "mlp_beta")
    assert not hasattr(DualHeadDecoder(WIDTH, global_offset=False), "mlp_beta")


def test_identity_table_has_one_row_per_selected_gene() -> None:
    """Sized to the gene axis, the table has no row that cannot learn."""
    encoder = _encoder()
    assert encoder.lifter.gene_emb.weight.shape == (N_GENES, WIDTH)


def test_a_seeded_generator_reproduces_the_mask() -> None:
    """The same generator seed draws the same mask; the global stream does not."""
    masker = MaskedMultiModalMasker(0.3, 0.3)
    values = torch.randn(4, 50, 3)
    first, _ = masker(values, generator=torch.Generator().manual_seed(7))
    second, _ = masker(values, generator=torch.Generator().manual_seed(7))

    assert torch.equal(first, second)


# --------------------------------------------------------------------------
# Loss and trainer
# --------------------------------------------------------------------------
def test_per_modality_averaging_weighs_each_omics_equally() -> None:
    """An omics masked ten times as often no longer dominates the loss."""
    targets = torch.zeros(1, 20, 3)
    prediction = torch.zeros(1, 20, 3)
    prediction[..., 0] = 1.0  # every expression entry is off by one
    mask = torch.zeros(1, 20, 3, dtype=torch.bool)
    mask[0, :, 0] = True  # twenty expression entries
    mask[0, :2, 1] = True  # two copy-number entries, predicted exactly

    pooled, _, _ = masked_huber_dual(prediction, prediction, targets, mask)
    averaged, _, _ = masked_huber_dual(
        prediction, prediction, targets, mask, per_modality=True
    )

    assert pooled.item() == pytest.approx(2 * 0.5 * 20 / 22)
    assert averaged.item() == pytest.approx(2 * 0.5)


def _trainer(dataset: TensorDataset, **kwargs) -> MaskedReconstructionTrainer:
    """Build a trainer over a tiny encoder and a fixed graph."""
    loader = DataLoader(dataset, batch_size=4)
    return MaskedReconstructionTrainer(
        _encoder(),
        loader,
        loader,
        torch.device("cpu"),
        _spd(),
        torch.randn(N_GENES, 4),
        **kwargs,
    )


def test_imputed_entries_are_never_scored() -> None:
    """Entries marked unobserved leave the loss untouched whatever they hold."""
    values = torch.randn(8, N_GENES, 3)
    observed = torch.ones_like(values, dtype=torch.bool)
    observed[:, :, 2] = False
    mask = torch.ones_like(observed)
    trainer = _trainer(
        TensorDataset(*(values[..., k] for k in range(3)), observed, mask)
    )
    first = trainer.validate_epoch()
    values[..., 2] = 100.0
    trainer.val_loader = DataLoader(
        TensorDataset(*(values[..., k] for k in range(3)), observed, mask),
        batch_size=4,
    )
    second = trainer.validate_epoch()

    assert math.isnan(first.per_modality[2])
    assert first.per_modality[:2] == pytest.approx(second.per_modality[:2], rel=0.2)


def test_a_fixed_monitoring_mask_makes_validation_deterministic() -> None:
    """With the mask carried in the batch, two validations agree exactly."""
    values = torch.randn(8, N_GENES, 3)
    observed = torch.ones_like(values, dtype=torch.bool)
    mask, _ = MaskedMultiModalMasker()(
        values, generator=torch.Generator().manual_seed(0)
    )
    trainer = _trainer(
        TensorDataset(*(values[..., k] for k in range(3)), observed, mask)
    )

    assert trainer.validate_epoch().val == trainer.validate_epoch().val


def test_gates_train_at_their_own_multiple_of_the_learning_rate() -> None:
    """The gate group's rate is the base rate times the multiplier."""
    dataset = TensorDataset(*(torch.randn(8, N_GENES) for _ in range(3)))
    trainer = _trainer(dataset, lr=1e-3, gate_lr_multiplier=10.0)

    rates = sorted(group["lr"] for group in trainer.optimizer.param_groups)
    assert rates == pytest.approx([1e-3, 1e-2])


def test_learning_rate_warms_up_then_follows_a_cosine() -> None:
    """Linear warm-up to the base rate, then cosine decay to the floor."""
    dataset = TensorDataset(*(torch.randn(8, N_GENES) for _ in range(3)))
    trainer = _trainer(
        dataset, lr=1e-3, min_lr=1e-5, warmup_epochs=4, lr_schedule="cosine"
    )

    factors = [trainer._learning_rate_factor(epoch, 14) for epoch in range(14)]
    assert factors[:4] == pytest.approx([0.25, 0.5, 0.75, 1.0])
    assert factors[9] == pytest.approx(0.01 + 0.99 * 0.5, rel=1e-6)
    assert all(a >= b for a, b in pairwise(factors[3:]))
    with pytest.raises(ValueError, match="lr_schedule"):
        _trainer(dataset, lr_schedule="step")


# --------------------------------------------------------------------------
# Preprocessing and splits
# --------------------------------------------------------------------------
def test_standardised_values_are_clipped_and_imputed_ones_flagged() -> None:
    """|z| never exceeds the clip, and the observed mask marks imputed entries."""
    rng = np.random.default_rng(0)
    features = rng.normal(size=(200, 12))
    features[0, 4] = 1e4  # an extreme copy-number value
    features[1, 0] = np.nan
    preprocessor = MultiOmicsTransformer(
        n_genes=4, gene_names=list("ABCD"), top_k=4, rna_scale="log", clip_z=5.0
    ).fit(features)

    transformed = preprocessor.transform(features)
    observed = preprocessor.observed_mask(features)

    assert np.abs(transformed).max() <= 5.0
    assert observed.shape == transformed.shape
    assert observed.sum() == observed.size - 1


def _cohort(n_patients: int = 120) -> OmicsData:
    """Return a synthetic cohort with two labels and an unlabelled group."""
    rng = np.random.default_rng(1)
    labels = np.repeat([0, 1, 2], n_patients // 3)
    return OmicsData(
        X=np.abs(rng.normal(size=(n_patients, 3 * 30))) * 5,
        y=labels,
        patient_ids=[f"P{i:03d}" for i in range(n_patients)],
        gene_names=[f"G{i}" for i in range(30)],
        label_map={"A": 0, "B": 1, "unlabelled": 2},
        inverse_label_map={0: "A", 1: "B", 2: "unlabelled"},
    )


def test_patient_splits_are_disjoint_and_independent_of_the_training_seed() -> None:
    """Three disjoint splits cover the cohort; the split seed alone fixes them."""
    data = _cohort()
    splits = split_pretraining_patients(data, 0.12, 0.10, seed=42)
    again = split_pretraining_patients(data, 0.12, 0.10, seed=42)

    together = np.concatenate([splits.train, splits.monitor, splits.heldout])
    assert sorted(together) == list(range(data.n_patients))
    assert len(splits.heldout) == 12
    assert np.array_equal(splits.heldout, again.heldout)


def test_heldout_reconstruction_reports_both_tasks_against_the_baseline() -> None:
    """Every omics gets a masked and a hidden-omics row per head."""
    inputs = torch.randn(100, N_GENES, 3)
    splits = PatientSplits(np.arange(30), np.arange(30, 40), np.arange(40, 100))
    rows = heldout_reconstruction(
        _encoder().eval(),
        inputs,
        torch.ones_like(inputs, dtype=torch.bool),
        splits,
        torch.randn(N_GENES, 4),
        _spd(),
        None,
        torch.device("cpu"),
    )
    tasks = {(row["task"], row["omics"], row["head"]) for row in rows}

    assert len(tasks) == 2 * 3 * 2
    assert all(row["gene_mean_baseline"] > 0 for row in rows)


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------
def test_a_pretraining_run_writes_every_artifact(tmp_path) -> None:
    """A two-epoch run writes weights, read-outs, splits and reconstruction."""
    data = _cohort()
    (tmp_path / "curated.txt").write_text("G0\nG1\n", encoding="utf-8")
    edges = tmp_path / "edges.tsv"
    pd.DataFrame(
        {
            "gene_a": ["G0", "G1", "G2"],
            "gene_b": ["G1", "G2", "G3"],
            "score": [0.9, 0.5, 0.7],
        }
    ).to_csv(edges, sep="\t", index=False)

    config = ExperimentConfig(name="synthetic", results_dir=str(tmp_path))
    config.data.curated_genes_file = str(tmp_path / "curated.txt")
    config.data.cohort_rule = "synthetic"
    config.preprocess.top_k = 8
    config.preprocess.clip_z = 5.0
    config.graph.edges_file = str(edges)
    config.graph.weighted_pe = True
    config.graph.pe_dim = 4
    config.graph.max_distance = 4
    config.model.d = WIDTH
    config.model.mini_heads = 2
    config.model.global_heads = 2
    config.model.lambda_gate = True
    config.model.fusion_type = "gated"
    config.model.global_offset = False
    config.model.isolated_token = True
    config.train.max_epochs = 2
    config.train.batch_size = 16
    config.train.warmup_epochs = 1
    config.train.lr_schedule = "cosine"
    config.train.per_modality_loss = True
    config.train.heldout_frac = 0.1
    config.train.n_readout_masks = 2

    bundle = run_pretraining(config, data, device=torch.device("cpu"))
    root = tmp_path / "synthetic"

    assert set(bundle.readouts) == set(READOUTS)
    for name in READOUTS:
        assert (root / "embeddings" / f"{name}.parquet").exists()
    for name in (
        "encoder_frozen.pt",
        "graph_tensors.pt",
        "probe_inputs.pt",
        "C_all.parquet",
        "training_history.csv",
        "config.yaml",
    ):
        assert (root / name).exists(), name
    provenance = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["splits"]["heldout"]["n"] == 12
    assert provenance["cohort_rule"] == "synthetic"
    reconstruction = json.loads(
        (root / "heldout_reconstruction.json").read_text(encoding="utf-8")
    )
    assert len(reconstruction["rows"]) == 12


# --------------------------------------------------------------------------
# Tested encoder changes: masking, identity modulation, summary tokens,
# fusion residual, covariance penalty
# --------------------------------------------------------------------------
def test_omics_level_masking_hides_a_whole_omics_for_some_patients() -> None:
    """With probability one, every patient has one omics hidden at every gene."""
    masker = MaskedMultiModalMasker(0.0, 0.0, omics_mask_prob=1.0)
    mask, _ = masker(torch.randn(16, 40, 3))

    whole = mask.all(dim=1)
    assert torch.all(whole.sum(dim=-1) == 1)


def test_neighbourhood_masks_hide_a_gene_with_its_neighbours() -> None:
    """On a graph of disjoint pairs, a masked gene's partner is masked too."""
    n_genes = 40
    adjacency = torch.zeros(n_genes, n_genes)
    for i in range(0, n_genes, 2):
        adjacency[i, i + 1] = adjacency[i + 1, i] = 1.0
    masker = MaskedMultiModalMasker(0.2, 0.0, neighbourhood_share=1.0)
    masker.set_neighbourhoods(adjacency)

    genes = masker(torch.randn(64, n_genes, 3))[0].all(dim=-1)
    pairs = genes.view(64, n_genes // 2, 2)
    partner_masked = (pairs[..., 0] & pairs[..., 1]).sum() * 2 / genes.sum()

    # Uniform masking would hide a gene's partner 20% of the time; seeds whose
    # pairs collide leave a small remainder filled gene by gene.
    assert partner_masked > 0.75
    assert genes.sum(dim=1).float().mean() == pytest.approx(0.2 * n_genes, abs=1)


def test_masking_without_the_new_options_is_unchanged() -> None:
    """The default masker draws exactly what it drew before the options existed."""
    values = torch.randn(4, 30, 3)
    plain, _ = MaskedMultiModalMasker()(
        values, generator=torch.Generator().manual_seed(3)
    )
    masker = MaskedMultiModalMasker()
    masker.set_neighbourhoods(torch.ones(30, 30))
    with_graph, _ = masker(values, generator=torch.Generator().manual_seed(3))

    assert torch.equal(plain, with_graph)


@pytest.mark.parametrize(
    "options",
    [
        {"identity_modulation": True},
        {"fusion_residual": True},
        {"n_summary_tokens": 4, "global_head": "cross_attention"},
        {"n_summary_tokens": 3, "global_head": "film"},
    ],
)
def test_zero_initialised_options_start_where_the_plain_encoder_starts(options) -> None:
    """Modulation and the residual start at zero; summary tokens widen the embedding."""
    plain = _encoder().eval()
    changed = _encoder(**options).eval()
    own = changed.state_dict()
    changed.load_state_dict(
        {
            k: v
            for k, v in plain.state_dict().items()
            if k in own and own[k].shape == v.shape
        },
        strict=False,
    )
    x = [torch.randn(2, N_GENES) for _ in range(3)]
    pe = torch.randn(N_GENES, 4)
    out = changed(*x, pe, _spd(), None, mask=True)
    k = options.get("n_summary_tokens", 1)

    assert out["c"].shape == (2, k * WIDTH)
    assert out["xhat_g"].shape == (2, N_GENES, 3)
    if k == 1:
        torch.testing.assert_close(
            out["xhat_l"],
            plain(*x, pe, _spd(), None, mask=True, mask_bool=out["mask_bool"])[
                "xhat_l"
            ],
        )


def test_the_decorrelation_penalty_ignores_scale_and_sees_correlation() -> None:
    """Independent columns score near zero; duplicates and their shrunk copy not."""
    from mogformer.training import embedding_correlation_penalty

    independent = torch.randn(4000, 4)
    duplicated = torch.cat([independent[:, :2], independent[:, :2]], dim=1)

    assert embedding_correlation_penalty(independent) < 0.01
    assert embedding_correlation_penalty(duplicated) > 0.4
    torch.testing.assert_close(
        embedding_correlation_penalty(duplicated),
        embedding_correlation_penalty(duplicated * 0.01),
        rtol=1e-3,
        atol=1e-4,
    )


def test_uniformity_is_lower_for_spread_embeddings_and_reads_the_bank() -> None:
    """Spread points score lower; bank rows are partners without gradient."""
    from mogformer.training import uniformity_loss

    spread = torch.randn(64, 16)
    clustered = torch.randn(1, 16) + 0.05 * torch.randn(64, 16)
    assert uniformity_loss(spread) < uniformity_loss(clustered) - 1.0

    batch = torch.randn(8, 16, requires_grad=True)
    bank = torch.nn.functional.normalize(torch.randn(32, 16), dim=1)
    with_bank = uniformity_loss(batch, bank)
    with_bank.backward()
    assert batch.grad is not None
    assert with_bank != uniformity_loss(batch.detach())


def test_vicreg_terms_see_shrunken_and_duplicated_dimensions() -> None:
    """Shrinking columns costs variance; duplicating them costs covariance."""
    from mogformer.training import vicreg_covariance, vicreg_variance

    independent = torch.randn(4000, 8)
    assert vicreg_variance(independent) < 0.02
    assert vicreg_variance(0.1 * independent) > 0.85
    assert vicreg_covariance(independent) < 0.01
    duplicated = torch.cat([independent[:, :4], independent[:, :4]], dim=1)
    assert vicreg_covariance(duplicated) > 0.4


def test_rank_me_counts_the_directions_in_use() -> None:
    """Isotropic data uses every direction; rank-one data uses one."""
    from mogformer.training import rank_me

    assert rank_me(torch.randn(2000, 10)) > 9.5
    assert rank_me(torch.randn(2000, 1) @ torch.randn(1, 10)) < 1.1


def test_the_memory_bank_keeps_the_newest_rows_detached() -> None:
    """Past capacity the oldest rows go first, and none carry gradient."""
    from mogformer.training import EmbeddingMemoryBank

    bank = EmbeddingMemoryBank(5)
    assert bank.values is None
    for start in (0, 3, 6):
        bank.push(torch.arange(start, start + 3, dtype=torch.float32)[:, None])
    assert bank.values is not None
    assert bank.values.squeeze(1).tolist() == [4.0, 5.0, 6.0, 7.0, 8.0]
    bank.push(torch.ones(2, 1, requires_grad=True))
    assert not bank.values.requires_grad


def test_regularisers_train_their_expander_and_fill_their_banks() -> None:
    """The expander is optimised and the epoch's summaries reach the banks."""
    dataset = TensorDataset(*(torch.randn(8, N_GENES) for _ in range(3)))
    trainer = _trainer(dataset, uniformity_weight=0.1, vicreg_weight=0.1)
    assert trainer.expander is not None
    before = [p.detach().clone() for p in trainer.expander.parameters()]

    assert math.isfinite(trainer.train_epoch())
    after = list(trainer.expander.parameters())
    assert any(not torch.equal(a, b) for a, b in zip(before, after, strict=True))
    assert trainer.uniformity_bank.values is not None
    assert trainer.expanded_bank.values is not None
    assert len(trainer.uniformity_bank.values) == 8


@pytest.mark.parametrize(
    ("selection", "expected_epoch"), [("loss", 6), ("loss_then_rank", 3)]
)
def test_rank_selection_keeps_the_most_spread_epoch_near_the_best_loss(
    selection: str, expected_epoch: int
) -> None:
    """Within 1% of the best loss the highest RankMe wins, and only there."""
    from mogformer.training.trainer import ValidationScores

    dataset = TensorDataset(*(torch.randn(8, N_GENES) for _ in range(3)))
    trainer = _trainer(dataset, selection=selection, selection_tolerance=0.01)
    script = iter([(1.0, 3), (0.5, 4), (0.502, 9), (0.501, 6), (0.6, 20), (0.499, 5)])
    trainer.train_epoch = lambda: 0.0  # type: ignore[method-assign]
    trainer.validate_epoch = lambda: ValidationScores(  # type: ignore[method-assign]
        *(lambda v, r: (v, v, v, r, r, [v] * 3, [1.0] * 3))(*next(script))
    )

    trainer.fit_early(max_epochs=6, patience=10, log_every=100)

    assert trainer.selected_epoch == expected_epoch
    if selection == "loss_then_rank":
        assert sorted(c[2] for c in trainer._candidates) == [3, 4, 6]


# --------------------------------------------------------------------------
# Gene programs
# --------------------------------------------------------------------------
def test_coexpression_programs_recover_co_varying_gene_blocks() -> None:
    """Genes driven by the same latent factor share a program, numbered by size."""
    from mogformer.data.gene_programs import fit_coexpression_programs

    factors = torch.randn(200, 3)
    blocks = [0] * 6 + [1] * 4 + [2] * 2
    values = torch.stack(
        [factors[:, b : b + 1].expand(-1, 3) for b in blocks], dim=1
    ) + 0.1 * torch.randn(200, 12, 3)

    programs = fit_coexpression_programs(values.numpy(), 3)

    assert programs.tolist() == blocks


def test_a_program_reads_only_its_member_genes() -> None:
    """Changing a gene moves its own program's embedding and no other."""
    from mogformer.models.layers import ProgramReadout

    readout = ProgramReadout(d=WIDTH, n_genes=6, n_programs=2, width=4).eval()
    readout.set_assignment(torch.tensor([0, 0, 1, 1, 1, 0]))
    genes = torch.randn(3, 6, WIDTH)
    moved = genes.clone()
    moved[:, 3] += 5.0

    before, after = readout(genes), readout(moved)

    torch.testing.assert_close(before[:, 0], after[:, 0])
    assert not torch.allclose(before[:, 1], after[:, 1])
    with pytest.raises(ValueError, match="needs a gene"):
        readout.set_assignment(torch.zeros(6, dtype=torch.long))


def test_the_scoped_global_head_reads_each_gene_from_its_own_program() -> None:
    """A program's embedding reaches the global predictions of its genes only."""
    decoder = DualHeadDecoder(d=WIDTH, global_offset=True, program_width=4).eval()
    assignment = torch.tensor([0, 1, 1, 0, 1])
    programs = torch.randn(2, 2, 4)
    moved = programs.clone()
    moved[:, 1] += 3.0
    arguments = (
        torch.randn(2, WIDTH),
        torch.randn(2, 6, WIDTH),
        torch.randn(5, WIDTH),
        torch.randn(3, WIDTH),
    )

    before, _ = decoder(*arguments, programs=programs, assignment=assignment)
    after, _ = decoder(*arguments, programs=moved, assignment=assignment)

    changed = (before - after).abs().amax(dim=(0, 2)) > 1e-6
    assert changed.tolist() == [False, True, True, False, True]
    with pytest.raises(ValueError, match="film head only"):
        DualHeadDecoder(d=WIDTH, global_head="cross_attention", program_width=4)


def test_a_program_encoder_embeds_into_its_blocks_and_saves_its_programs() -> None:
    """The embedding is K blocks wide, and the assignment travels with the weights."""
    model = _encoder(n_programs=3, program_width=4).eval()
    assignment = torch.arange(N_GENES) % 3
    model.program_readout.set_assignment(assignment.flip(0))
    values = torch.randn(2, N_GENES)

    out = model(values, values, values, torch.randn(N_GENES, 4), _spd(), mask=False)
    masked = model(values, values, values, torch.randn(N_GENES, 4), _spd())
    reloaded = _encoder(n_programs=3, program_width=4)
    reloaded.load_state_dict(model.state_dict())

    assert out["c"].shape == (2, 12) == masked["c"].shape
    assert model.embedding_width == 12
    torch.testing.assert_close(reloaded.program_readout.assignment, assignment.flip(0))


def test_programs_reading_fused_tokens_ignore_genes_outside_the_program() -> None:
    """Read before the inter-gene transformer, a block sees only its own genes."""
    model = _encoder(n_programs=2, program_width=4, program_input="fused").eval()
    model.program_readout.set_assignment(torch.tensor([0, 0, 0, 1, 1, 1]))
    pe = torch.randn(N_GENES, 4)
    values = torch.randn(2, N_GENES)
    moved = values.clone()
    moved[:, 4] += 3.0

    before = model(values, values, values, pe, _spd(), mask=False)["c"]
    after = model(moved, moved, moved, pe, _spd(), mask=False)["c"]

    torch.testing.assert_close(before[:, :4], after[:, :4])
    assert not torch.allclose(before[:, 4:], after[:, 4:])


# --------------------------------------------------------------------------
# Cross-omics contrastive views
# --------------------------------------------------------------------------
def test_a_forced_omics_is_hidden_for_every_gene_of_its_patient() -> None:
    """Each patient loses exactly the omics it was given, at every gene."""
    masker = MaskedMultiModalMasker(omics_mask_prob=1.0)
    mask, _ = masker(torch.randn(3, N_GENES, 3), hidden_omics=torch.tensor([0, 2, 1]))

    assert mask[0, :, 0].all() and mask[1, :, 2].all() and mask[2, :, 1].all()


def test_a_patient_given_no_omics_keeps_every_omics_visible_somewhere() -> None:
    """-1 forces no whole omics, and the random omics-level draw is skipped."""
    masker = MaskedMultiModalMasker(mask_gene_frac=0.0, mask_modality_frac=0.0)
    mask, _ = masker(torch.randn(2, N_GENES, 3), hidden_omics=torch.tensor([-1, 1]))

    assert not mask[0].any()
    assert mask[1, :, 1].all() and not mask[1, :, [0, 2]].any()


def test_info_nce_rewards_matching_views_and_reads_the_bank() -> None:
    """Matched views score low, shuffled ones high; bank rows add negatives."""
    from mogformer.training import info_nce

    views = torch.randn(16, 8)
    matched = info_nce(views, views + 0.01 * torch.randn(16, 8), temperature=0.1)
    shuffled = info_nce(views, views[torch.randperm(16)], temperature=0.1)
    bank = torch.nn.functional.normalize(torch.randn(64, 8), dim=1)

    assert matched < 0.1 < shuffled
    assert info_nce(views, views, bank, temperature=0.3) > info_nce(
        views, views, temperature=0.3
    )


def test_the_two_view_step_trains_its_head_and_logs_the_contrastive_loss() -> None:
    """Both views are encoded, the head learns, and the epoch's loss is kept."""
    dataset = TensorDataset(*(torch.randn(8, N_GENES) for _ in range(3)))
    trainer = _trainer(dataset, contrastive_weight=0.5)
    assert trainer.projector is not None
    before = [p.detach().clone() for p in trainer.projector.parameters()]

    assert math.isfinite(trainer.train_epoch())

    after = list(trainer.projector.parameters())
    assert any(not torch.equal(a, b) for a, b in zip(before, after, strict=True))
    assert len(trainer.history.contrastive) == 1
    assert trainer.contrastive_bank.values is not None
    assert len(trainer.contrastive_bank.values) == 16


def test_two_views_without_the_loss_train_no_head() -> None:
    """The control encodes two views but adds no contrastive term or head."""
    dataset = TensorDataset(*(torch.randn(8, N_GENES) for _ in range(3)))
    trainer = _trainer(dataset, two_views=True)

    assert trainer.two_views and trainer.projector is None
    assert math.isfinite(trainer.train_epoch())
    assert trainer.history.contrastive == []


def test_integrated_gradients_are_complete_and_find_the_driving_input() -> None:
    """Attributions sum to the score change; program attention stays in-program."""
    from mogformer.analysis.attribution import integrated_gradients

    model = _encoder(n_programs=2, program_width=4, program_input="fused").eval()
    model.program_readout.set_assignment(torch.tensor([0, 0, 0, 1, 1, 1]))
    inputs = torch.randn(5, N_GENES, 3)
    direction = torch.zeros(model.embedding_width)
    direction[:4] = 1.0

    attributions, gap = integrated_gradients(
        model,
        inputs,
        direction,
        torch.randn(N_GENES, 4),
        _spd(),
        None,
        torch.device("cpu"),
        steps=64,
        batch_size=2,
    )
    scale = np.abs(attributions.sum(axis=(1, 2))).max()

    assert attributions.shape == (5, N_GENES, 3)
    assert np.abs(gap).max() < 0.05 * scale + 1e-3
    assert np.abs(attributions[:, 3:]).max() < 1e-6
    weights = model.program_readout.attention_weights(torch.randn(2, N_GENES, WIDTH))
    assert weights.shape == (2, 2, N_GENES)
    assert torch.allclose(weights[:, 0, 3:], torch.zeros(2, 3))
