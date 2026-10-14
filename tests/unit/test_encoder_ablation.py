"""Tests for the leave-one-group-out summary of the encoder ablation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mogformer.experiments.encoder_ablation import run_encoder_ablation

TARGETS = ("PAM50", "ER status", "HER2 status")


def _evaluation(
    root: Path, name: str, runs: dict[str, tuple[float, float, float]]
) -> None:
    """Write a minimal suite output: probes, axes and reconstruction per run.

    Each run maps to (probe macro-F1, held-out loss ratio, participation ratio).
    """
    directory = root / name
    directory.mkdir(parents=True)
    many = len(runs) > 1
    probes, axes, reconstruction = [], [], []
    for run, (f1, loss, pr) in runs.items():
        for readout, shift in (("summary", 0.0), ("gene_mean", -0.05)):
            key = f"encoder_{readout}" + (f"__{run}" if many else "")
            probes += [
                {
                    "target": t,
                    "representation": key,
                    "task": "classification",
                    "nuisance": False,
                    "mean": f1 + shift,
                }
                for t in TARGETS
            ]
            probes.append(
                {
                    "target": "Site",
                    "representation": key,
                    "task": "classification",
                    "nuisance": True,
                    "mean": 0.1,
                }
            )
            axes.append({"representation": key, "participation_ratio": pr})
        for omics in ("rna", "cnv", "methy"):
            for head in ("local", "global"):
                reconstruction.append(
                    {
                        "run": run,
                        "task": "masked",
                        "omics": omics,
                        "head": head,
                        "ratio_to_baseline": loss,
                    }
                )
                reconstruction.append(
                    {
                        "run": run,
                        "task": "omics_hidden",
                        "omics": omics,
                        "head": head,
                        "ratio_to_baseline": 1.0,
                    }
                )
    pd.DataFrame(probes).to_csv(directory / "probe_summary.csv", index=False)
    pd.DataFrame(axes).to_csv(directory / "axes_summary.csv", index=False)
    pd.DataFrame(reconstruction).to_csv(directory / "reconstruction.csv", index=False)


def test_groups_are_judged_against_reference_seed_noise(tmp_path) -> None:
    """A large probe gain keeps a group; a change inside the noise does not."""
    _evaluation(
        tmp_path,
        "reference",
        {
            "seed42": (0.70, 0.62, 8.0),
            "seed43": (0.71, 0.63, 8.5),
            "seed44": (0.72, 0.61, 7.5),
        },
    )
    _evaluation(
        tmp_path, "enc_all_groups", {"enc_all_groups_seed42": (0.80, 0.55, 16.0)}
    )
    _evaluation(tmp_path, "enc_without_masking", {"r": (0.70, 0.62, 9.0)})
    _evaluation(tmp_path, "enc_without_fusion_residual", {"r": (0.80, 0.55, 16.2)})

    record = run_encoder_ablation(
        tmp_path,
        tmp_path / "out",
        "tcga",
        "reference",
        "enc",
        groups=("masking", "fusion_residual"),
    )

    assert record["verdicts"] == {"masking": "keep", "fusion_residual": "no difference"}
    assert (tmp_path / "out" / "encoder_card_tcga.md").exists()
    readouts = pd.read_csv(tmp_path / "out" / "ablation_readouts_tcga.csv")
    assert set(readouts["readout"]) == {"summary"}
