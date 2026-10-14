"""Per-patient hormone-receptor and HER2 status for TCGA-BRCA.

GDC stores each patient's receptor tests as rows of ``molecular_tests`` in its
follow-up table: one row per test, several tests per patient, ER and PR by
immunohistochemistry and HER2 by immunohistochemistry and FISH. This module
collapses those rows to one status per patient and receptor, following the
clinical reading order for HER2 (a FISH result decides; immunohistochemistry
decides only when no FISH exists; an equivocal immunohistochemistry without
FISH stays unknown). Where GDC has no usable status, the TCGA 2012 clinical
table fills the gap.

The statuses define populations and probe labels only; nothing here reaches a
model's inputs.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

#: Statuses a patient can end with. Only the first two are usable labels.
KNOWN_STATUSES: tuple[str, ...] = ("Positive", "Negative")

#: HER2/CEP17 FISH ratio at or above which HER2 is called amplified.
HER2_FISH_RATIO_THRESHOLD = 2.0

#: GDC gene symbols of the three receptor tests.
RECEPTOR_GENES: dict[str, str] = {"ESR1": "er", "PGR": "pr", "ERBB2": "her2"}

#: Columns of the TCGA 2012 clinical table holding the same three statuses.
LEGACY_COLUMNS: dict[str, str] = {
    "ER_STATUS": "er",
    "PR_STATUS": "pr",
    "HER2_STATUS": "her2",
}

_PATIENT = "cases.submitter_id"
_GENE = "molecular_tests.gene_symbol"
_RESULT = "molecular_tests.test_result"
_METHOD = "molecular_tests.molecular_analysis_method"
_VALUE = "molecular_tests.test_value"


def _single_status(results: Iterable[str]) -> str:
    """Collapse several test results into one status.

    Args:
        results: Results of the tests considered, e.g. ``"Positive"``.

    Returns:
        ``"Positive"`` or ``"Negative"`` when the definite results agree,
        ``"Discordant"`` when they disagree, and ``"Unknown"`` when there is
        no definite result.
    """
    definite = {result for result in results if result in KNOWN_STATUSES}
    if len(definite) == 1:
        return definite.pop()
    return "Discordant" if definite else "Unknown"


def _fish_result(result: str, value: object) -> str | None:
    """Read one FISH row as a status, from its result or its reported ratio.

    Args:
        result: The row's ``test_result``.
        value: The row's ``test_value``, the HER2/CEP17 ratio when reported.

    Returns:
        A status, or None when the row carries neither a result nor a ratio.
    """
    if result in KNOWN_STATUSES:
        return result
    try:
        ratio = float(str(value))
    except ValueError:
        return None
    if ratio != ratio:  # NaN
        return None
    return "Positive" if ratio >= HER2_FISH_RATIO_THRESHOLD else "Negative"


def her2_status(tests: pd.DataFrame) -> str:
    """Resolve one patient's HER2 status from their HER2 test rows.

    Args:
        tests: The patient's ``ERBB2`` rows from the GDC follow-up table.

    Returns:
        The FISH call when any FISH row is readable, otherwise the
        immunohistochemistry call; ``"Unknown"`` for an equivocal
        immunohistochemistry without FISH.
    """
    fish = tests[tests[_METHOD] == "FISH"]
    calls = [
        call
        for call in (
            _fish_result(row[_RESULT], row[_VALUE]) for _, row in fish.iterrows()
        )
        if call is not None
    ]
    if calls:
        return _single_status(calls)
    return _single_status(tests.loc[tests[_METHOD] == "IHC", _RESULT])


def gdc_receptor_status(follow_up: pd.DataFrame) -> pd.DataFrame:
    """Collapse GDC molecular tests to one status per patient and receptor.

    Args:
        follow_up: The GDC ``follow_up.tsv`` table, with ``'--`` already read
            as missing.

    Returns:
        One row per patient with an ``er``, ``pr`` and ``her2`` column; a
        patient without any test of a receptor is absent from that column.
    """
    tests = follow_up.dropna(subset=[_GENE])
    columns = {}
    for gene, name in RECEPTOR_GENES.items():
        rows = tests[tests[_GENE] == gene]
        if name == "her2":
            columns[name] = pd.Series(
                {
                    patient: her2_status(patient_tests)
                    for patient, patient_tests in rows.groupby(_PATIENT)
                },
                dtype=object,
            )
        else:
            columns[name] = rows.groupby(_PATIENT)[_RESULT].agg(_single_status)
    frame = pd.DataFrame(columns)
    frame.index.name = "patient_id"
    return frame


def resolve_receptor_status(
    follow_up_path: str | Path, legacy_path: str | Path | None = None
) -> pd.DataFrame:
    """Build the receptor table: GDC first, the TCGA 2012 table for the gaps.

    Args:
        follow_up_path: GDC ``follow_up.tsv`` of the TCGA-BRCA project.
        legacy_path: cBioPortal ``data_clinical_sample.txt`` of the TCGA 2012
            study, or None to use GDC alone.

    Returns:
        One row per patient with the status of each receptor, the source of
        each status (``gdc`` or ``tcga_2012``) and ``er_positive_her2_negative``,
        true only when both statuses are known and the patient is ER+/HER2−.
    """
    follow_up = pd.read_csv(
        follow_up_path, sep="\t", dtype=str, na_values=["'--"], low_memory=False
    )
    table = gdc_receptor_status(follow_up)
    for name in RECEPTOR_GENES.values():
        table[f"{name}_source"] = (
            table[name]
            .where(table[name].isin(KNOWN_STATUSES))
            .notna()
            .map({True: "gdc", False: ""})
        )

    if legacy_path is not None:
        legacy = pd.read_csv(legacy_path, sep="\t", comment="#").set_index("PATIENT_ID")
        legacy = legacy[list(LEGACY_COLUMNS)].rename(columns=LEGACY_COLUMNS)
        table = table.reindex(table.index.union(legacy.index))
        for name in RECEPTOR_GENES.values():
            missing = ~table[name].isin(KNOWN_STATUSES)
            fill = legacy[name].reindex(table.index)
            usable = missing & fill.isin(KNOWN_STATUSES)
            table.loc[usable, name] = fill[usable]
            table.loc[usable, f"{name}_source"] = "tcga_2012"
        table.index.name = "patient_id"

    for name in RECEPTOR_GENES.values():
        table[name] = table[name].fillna("Unknown")
        table[f"{name}_source"] = table[f"{name}_source"].fillna("")
    table["er_positive_her2_negative"] = (table["er"] == "Positive") & (
        table["her2"] == "Negative"
    )
    return table.sort_index()
