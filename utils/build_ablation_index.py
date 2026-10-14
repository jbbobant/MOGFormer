"""Join every archived run to its scores, so the ablation can be read at once.

The runs were produced over months of separate sessions and their only surviving
description is the directory name — there is no per-run ``config.yaml`` under
``results/REP_FOLDS/``. This tool decodes those names back into settings and
puts them beside the score and, crucially, beside the number of estimates that
score rests on.

That last column is the point. Read down the mean alone and the ablation says
one thing; read the mean next to ``n`` and it says something quite different,
because the best-looking runs are the ones that were never taken past a single
repeat.

Usage:
    uv run python -m utils.build_ablation_index
    uv run python -m utils.build_ablation_index --check
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Root holding the archived transformer runs.
RUNS_ROOT = REPO_ROOT / "results" / "REP_FOLDS"

#: Where the joined table is written.
INDEX_CSV = REPO_ROOT / "results" / "ablation_index.csv"
INDEX_MD = REPO_ROOT / "results" / "ablation_index.md"

#: Estimates a run needs before its interval is worth quoting. Five folds of a
#: single repeat is one partition, not a distribution over partitions.
FULL_PROTOCOL_N = 25

#: Directory-name fragments and the setting each one encodes. Order matters:
#: the longest match wins, so ``meth_M`` is read before ``meth``.
NAME_TOKENS: tuple[tuple[str, str, object], ...] = (
    ("ptemb16", "gene_embedding", "pretrained_rank16"),
    ("ptemb32", "gene_embedding", "pretrained_rank32"),
    ("ptemb64", "gene_embedding", "pretrained_rank64"),
    ("lgemb", "gene_embedding", "learned"),
    ("lemb", "gene_embedding", "learned"),
    ("meth_M", "methylation", "m_values"),
    ("met_M", "methylation", "m_values"),
    ("meth_mean", "methylation", "beta_mean_imputed"),
    ("met_b", "methylation", "beta"),
    ("mask_std", "unimodal_dropout_fill", "mask_token_std"),
    ("mask", "unimodal_dropout_fill", "mask_token"),
    ("gated", "fusion_type", "gated"),
    ("minitrans", "fusion_type", "attention"),
    ("dual", "attention_bias_mode", "dual"),
    ("inside", "attention_bias_mode", "inside"),
    ("lapl", "pe_method", "laplacian"),
    ("llr10000", "lambda_gate", True),
    ("logit_adjusted", "loss", "logit_adjusted"),
    ("grn", "use_grn", True),
    ("ppi", "use_ppi", True),
)

#: Fragments carrying a number, as a pattern and the setting it fills.
NUMERIC_TOKENS: tuple[tuple[str, str], ...] = (
    (r"^E(\d+)$", "max_epochs"),
    (r"^p(\d+)$", "patience"),
    (r"^fg(\d+)$", "focal_gamma"),
    (r"^gamma(\d+)$", "focal_gamma"),
    (r"^d(\d+)$", "max_distance"),
    (r"^lay(\d+)$", "global_layers"),
)

_SUMMARY_PATTERN = re.compile(
    r"macro-F1:\s*\*\*(?P<mean>[\d.]+)\*\*"
    r".*?\[(?P<lo>[\d.]+),\s*(?P<hi>[\d.]+)\]"
    r".*?NB CI \[(?P<nb_lo>[\d.]+),\s*(?P<nb_hi>[\d.]+)\]"
    r",\s*n=(?P<n>\d+)",
    re.DOTALL,
)


@dataclass
class RunRecord:
    """One archived run, joined to its score.

    Attributes:
        run: Directory name, the run's only original description.
        cohort: Class configuration the run used.
        group: Sub-directory grouping the run was archived under.
        macro_f1: Mean macro-F1 over the run's estimates.
        ci95_lo: Lower naive interval bound.
        ci95_hi: Upper naive interval bound.
        nb_ci95_lo: Lower Nadeau-Bengio corrected bound.
        nb_ci95_hi: Upper corrected bound.
        n: Estimates behind the mean.
        full_protocol: Whether ``n`` reached the pre-registered count.
        has_per_fold: Whether per-fold scores survive, so the run can still
            enter a paired test.
        settings: Settings decoded from the directory name.
        identical_to: Other runs whose per-fold scores are byte-identical to
            this one's. Two configurations cannot produce identical training
            outcomes, so such runs are copies whose origin cannot be
            attributed, and they are kept out of every ranking.
    """

    run: str
    cohort: str
    group: str
    macro_f1: float | None = None
    ci95_lo: float | None = None
    ci95_hi: float | None = None
    nb_ci95_lo: float | None = None
    nb_ci95_hi: float | None = None
    n: int | None = None
    full_protocol: bool = False
    has_per_fold: bool = False
    settings: dict[str, object] = field(default_factory=dict)
    identical_to: list[str] = field(default_factory=list)


def decode_run_name(name: str) -> dict[str, object]:
    """Recover the settings a run directory name encodes.

    Args:
        name: Directory name such as ``E300p40_inside_met_M_lgemb_fg2``.

    Returns:
        Mapping of setting to value for every fragment recognised. Fragments
        that are not recognised are collected under ``unparsed`` rather than
        dropped, so a silently misread name stays visible.
    """
    settings: dict[str, object] = {}
    remaining = name

    for token, key, value in NAME_TOKENS:
        if token in remaining:
            settings[key] = value
            remaining = remaining.replace(token, "", 1)

    unparsed: list[str] = []
    for fragment in (part for part in re.split(r"[_-]+", remaining) if part):
        for pattern, key in NUMERIC_TOKENS:
            match = re.match(pattern, fragment)
            if match:
                settings[key] = int(match.group(1))
                break
        else:
            # E300p40 arrives as one fragment; split the leading epoch count.
            joined = re.match(r"^E(\d+)p(\d+)$", fragment)
            if joined:
                settings["max_epochs"] = int(joined.group(1))
                settings["patience"] = int(joined.group(2))
            else:
                unparsed.append(fragment)

    if unparsed:
        settings["unparsed"] = "|".join(unparsed)
    return settings


def read_summary(path: Path) -> dict[str, float] | None:
    """Parse the headline numbers out of a run's Markdown summary.

    Args:
        path: A ``metrics_summary.md`` written by an archived run.

    Returns:
        Mapping of statistic to value, or None if the file does not carry the
        expected line.
    """
    match = _SUMMARY_PATTERN.search(path.read_text(encoding="utf-8"))
    if match is None:
        return None
    groups = match.groupdict()
    return {
        "macro_f1": float(groups["mean"]),
        "ci95_lo": float(groups["lo"]),
        "ci95_hi": float(groups["hi"]),
        "nb_ci95_lo": float(groups["nb_lo"]),
        "nb_ci95_hi": float(groups["nb_hi"]),
        "n": float(groups["n"]),
    }


def collect_runs(root: Path = RUNS_ROOT) -> list[RunRecord]:
    """Walk the archive and build one record per run.

    Args:
        root: Directory holding the archived runs.

    Returns:
        Records sorted by mean macro-F1, best first.
    """
    records: list[RunRecord] = []
    for summary_path in sorted(root.rglob("metrics_summary.md")):
        directory = summary_path.parent
        parts = directory.relative_to(root).parts
        record = RunRecord(
            run=parts[-1],
            cohort=parts[0] if parts else "",
            group="/".join(parts[1:-1]),
            has_per_fold=(directory / "metrics_per_fold.csv").exists(),
            settings=decode_run_name(parts[-1]),
        )
        summary = read_summary(summary_path)
        if summary is not None:
            record.macro_f1 = summary["macro_f1"]
            record.ci95_lo = summary["ci95_lo"]
            record.ci95_hi = summary["ci95_hi"]
            record.nb_ci95_lo = summary["nb_ci95_lo"]
            record.nb_ci95_hi = summary["nb_ci95_hi"]
            record.n = int(summary["n"])
            record.full_protocol = record.n >= FULL_PROTOCOL_N
        records.append(record)

    mark_identical_runs(records, root)
    return sorted(records, key=lambda record: (-(record.macro_f1 or 0.0), record.run))


def mark_identical_runs(records: Sequence[RunRecord], root: Path) -> None:
    """Flag runs whose per-fold score files are byte-identical.

    Args:
        records: Runs to check, updated in place.
        root: Directory holding the archived runs.
    """
    by_digest: dict[str, list[RunRecord]] = {}
    for record in records:
        if not record.has_per_fold:
            continue
        directory = root / record.cohort / record.group / record.run
        digest = hashlib.sha256(
            (directory / "metrics_per_fold.csv").read_bytes()
        ).hexdigest()
        by_digest.setdefault(digest, []).append(record)

    for group in by_digest.values():
        if len(group) < 2:
            continue
        for record in group:
            record.identical_to = sorted(
                f"{other.cohort}/{other.run}" for other in group if other is not record
            )


def to_frame(records: Sequence[RunRecord]) -> pd.DataFrame:
    """Flatten records into a table, one column per decoded setting.

    Args:
        records: Runs to tabulate.

    Returns:
        Frame with the score columns first and the settings after them.
    """
    rows = []
    for record in records:
        row = asdict(record)
        settings = row.pop("settings")
        row["identical_to"] = "; ".join(row["identical_to"])
        rows.append({**row, **{f"set_{key}": v for key, v in settings.items()}})
    return pd.DataFrame(rows)


def render_index(frame: pd.DataFrame) -> str:
    """Render the index as Markdown, with the protocol caveat stated first.

    Args:
        frame: Table from :func:`to_frame`.

    Returns:
        A Markdown document.
    """
    attributable = frame[frame["identical_to"] == ""]
    identical = frame[frame["identical_to"] != ""]
    full = attributable[attributable["full_protocol"]]
    partial = attributable[~attributable["full_protocol"]]
    lines = [
        "# Ablation index",
        "",
        f"{len(frame)} archived transformer runs, decoded from their directory",
        "names and joined to their scores. Generated by",
        "`uv run python -m utils.build_ablation_index`; do not edit by hand.",
        "",
        "## Read the estimate count first",
        "",
        f"Only {len(full)} of {len(attributable)} attributable runs reached the "
        f"pre-registered {FULL_PROTOCOL_N} estimates. The remaining "
        f"{len(partial)} rest on one repeat or two, which is a single "
        "partition rather than a distribution over partitions, so their "
        "intervals are not comparable to the full-protocol ones and their "
        "ranking is largely noise.",
        "",
    ]

    for title, subset in (
        (f"Full protocol (n = {FULL_PROTOCOL_N})", full),
        ("Partial protocol", partial),
    ):
        lines += [f"## {title}", ""]
        if subset.empty:
            lines += ["_none_", ""]
            continue
        lines += [
            "| run | cohort | macro-F1 | NB 95% CI | n | per-fold |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for _, row in subset.iterrows():
            interval = (
                f"[{row['nb_ci95_lo']:.3f}, {row['nb_ci95_hi']:.3f}]"
                if pd.notna(row["nb_ci95_lo"])
                else "n/a"
            )
            mean = f"{row['macro_f1']:.4f}" if pd.notna(row["macro_f1"]) else "n/a"
            count = "" if pd.isna(row["n"]) else str(int(row["n"]))
            lines.append(
                f"| `{row['run']}` | {row['cohort']} | {mean} | {interval} | "
                f"{count} | {'yes' if row['has_per_fold'] else 'no'} |"
            )
        lines.append("")

    if not identical.empty:
        lines += [
            "## Not attributable (byte-identical outputs)",
            "",
            "These runs have byte-identical per-fold scores. Two configurations "
            "cannot produce identical training outcomes, so all but one of each "
            "group are copies, and the repository does not record which "
            "configuration produced the numbers. They are excluded from both "
            "tables above.",
            "",
            "| run | cohort | macro-F1 | n | identical to |",
            "| --- | --- | --- | --- | --- |",
        ]
        for _, row in identical.iterrows():
            mean = f"{row['macro_f1']:.4f}" if pd.notna(row["macro_f1"]) else "n/a"
            count = "" if pd.isna(row["n"]) else str(int(row["n"]))
            lines.append(
                f"| `{row['run']}` | {row['cohort']} | {mean} | {count} | "
                f"{row['identical_to']} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Build the ablation index.

    Args:
        argv: Command-line arguments, or None for ``sys.argv``.

    Returns:
        Process exit status. Under ``--check``, non-zero when the committed
        index is stale.
    """
    parser = argparse.ArgumentParser(description="Build the ablation index.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the committed index differs from a fresh build",
    )
    args = parser.parse_args(argv)

    records = collect_runs()
    if not records:
        print(f"no runs found under {RUNS_ROOT}", file=sys.stderr)
        return 1

    frame = to_frame(records)
    markdown = render_index(frame)

    if args.check:
        if not INDEX_MD.exists() or INDEX_MD.read_text(encoding="utf-8") != markdown:
            print("ablation index is stale; re-run without --check", file=sys.stderr)
            return 1
        return 0

    INDEX_CSV.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(INDEX_CSV, index=False)
    INDEX_MD.write_text(markdown, encoding="utf-8")
    print(f"indexed {len(records)} runs -> {INDEX_MD.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
