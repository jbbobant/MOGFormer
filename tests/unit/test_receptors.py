"""Tests for collapsing GDC receptor tests to one status per patient."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mogformer.data.receptors import gdc_receptor_status, resolve_receptor_status


def _tests(rows: list[tuple[str, str, str, str, str]]) -> pd.DataFrame:
    """Build a follow-up table from (patient, gene, method, result, value)."""
    return pd.DataFrame(
        rows,
        columns=[
            "cases.submitter_id",
            "molecular_tests.gene_symbol",
            "molecular_tests.molecular_analysis_method",
            "molecular_tests.test_result",
            "molecular_tests.test_value",
        ],
    )


def test_fish_decides_her2_over_immunohistochemistry() -> None:
    """A FISH call overrides IHC; a reported ratio is read against 2.0."""
    table = gdc_receptor_status(
        _tests(
            [
                ("P1", "ERBB2", "IHC", "Equivocal", None),
                ("P1", "ERBB2", "FISH", "Positive", None),
                ("P2", "ERBB2", "IHC", "Positive", None),
                ("P2", "ERBB2", "FISH", "Test Value Reported", "1.3"),
                ("P3", "ERBB2", "FISH", "Copy Number Reported", "2.4"),
            ]
        )
    )

    assert table.loc["P1", "her2"] == "Positive"
    assert table.loc["P2", "her2"] == "Negative"
    assert table.loc["P3", "her2"] == "Positive"


def test_equivocal_immunohistochemistry_without_fish_stays_unknown() -> None:
    """Only definite results make a status; disagreement is flagged."""
    table = gdc_receptor_status(
        _tests(
            [
                ("P1", "ERBB2", "IHC", "Equivocal", None),
                ("P2", "ESR1", "IHC", "Positive", None),
                ("P2", "ESR1", "IHC", "Negative", None),
                ("P3", "ESR1", "IHC", "Positive", None),
                ("P3", "ESR1", "IHC", "Equivocal", None),
            ]
        )
    )

    assert table.loc["P1", "her2"] == "Unknown"
    assert table.loc["P2", "er"] == "Discordant"
    assert table.loc["P3", "er"] == "Positive"


def test_the_2012_table_fills_only_the_gaps(tmp_path: Path) -> None:
    """GDC wins where it is definite; the legacy table fills the rest."""
    follow_up = _tests(
        [
            ("P1", "ESR1", "IHC", "Positive", None),
            ("P1", "ERBB2", "IHC", "Equivocal", None),
        ]
    )
    follow_up_path = tmp_path / "follow_up.tsv"
    follow_up.to_csv(follow_up_path, sep="\t", index=False)
    legacy_path = tmp_path / "legacy.txt"
    legacy_path.write_text(
        "#header\nPATIENT_ID\tER_STATUS\tPR_STATUS\tHER2_STATUS\n"
        "P1\tNegative\tPositive\tNegative\n"
        "P2\tPositive\tPositive\tPositive\n",
        encoding="utf-8",
    )

    table = resolve_receptor_status(follow_up_path, legacy_path)

    assert table.loc["P1", "er"] == "Positive"
    assert table.loc["P1", "er_source"] == "gdc"
    assert table.loc["P1", "her2"] == "Negative"
    assert table.loc["P1", "her2_source"] == "tcga_2012"
    assert bool(table.loc["P1", "er_positive_her2_negative"])
    assert table.loc["P2", "her2"] == "Positive"
    assert not bool(table.loc["P2", "er_positive_her2_negative"])
