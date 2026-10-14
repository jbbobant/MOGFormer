"""Point a configuration at wherever the data happens to live.

A configuration in ``configs/`` names paths to the raw inputs, while an experiment
reads the cohort bundle that ``utils.build_cohort_bundle`` writes. Rewriting
those paths by hand per run is how two runs end up reading different things.

So the rebase happens once, here, from a declared layout — and the rebased
configuration is written next to the results, which makes every run
self-describing: what it read, where from, and on what hardware.
"""

from __future__ import annotations

import json
import platform
import sys
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mogformer.config import ExperimentConfig

#: Cohort archive used unless a run asks for another representation.
DEFAULT_COHORT_FILE = "cohort_beta.npz"


@dataclass(frozen=True)
class BundleLayout:
    """Where each input sits inside a cohort bundle.

    The names are fixed by :mod:`utils.build_cohort_bundle`, which writes the
    bundle. Keeping them in one dataclass means a rename breaks loudly in one
    place rather than silently in whichever run happened to read that file.

    Attributes:
        root: Directory holding the bundle.
        cohort_file: Cohort archive to read.
        folds_file: Shared partition.
        edges_file: Interaction edges already reduced to the gene universe.
        grn_file: Signed regulatory edges.
        curated_genes_file: Genes force-included after variance ranking.
        pam50_genes_file: The label-defining panel.
        embeddings_file: Frozen gene vectors.
        encoders_dir: Archived self-supervised runs, one subdirectory per
            cohort, that the trans probe re-measures from.
    """

    root: Path
    cohort_file: str = DEFAULT_COHORT_FILE
    folds_file: str = "folds.json"
    edges_file: str = "string_edges.tsv"
    grn_file: str = "collectri_signed_edges.tsv"
    curated_genes_file: str = "curated_genes.txt"
    pam50_genes_file: str = "pam50_genes.txt"
    embeddings_file: str = "gene_embeddings.npz"
    encoders_dir: str = "encoders"

    @property
    def cohort(self) -> Path:
        """Return the cohort archive path."""
        return self.root / self.cohort_file

    def require(self, with_folds: bool = True) -> BundleLayout:
        """Check that every file a run may read is present.

        Args:
            with_folds: Require the shared partition. Self-supervised runs
                read none.

        Returns:
            This layout, so the call can be chained.

        Raises:
            FileNotFoundError: If the bundle is incomplete, naming everything
                that is missing rather than only the first.
        """
        expected = {
            "cohort": self.cohort,
            "edges": self.root / self.edges_file,
            "curated genes": self.root / self.curated_genes_file,
        }
        if with_folds:
            expected["folds"] = self.root / self.folds_file
        missing = [
            f"{label} ({path})" for label, path in expected.items() if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "bundle at "
                f"{self.root} is missing: {', '.join(missing)}. Rebuild it with "
                "`uv run python -m utils.build_cohort_bundle`."
            )
        return self

    def encoder(self, cohort: str) -> Path:
        """Return the archived run directory for one cohort.

        Args:
            cohort: Cohort name, matching a subdirectory of ``encoders_dir``.

        Returns:
            The directory holding that cohort's frozen encoder.

        Raises:
            FileNotFoundError: If the bundle carries no encoder for it, naming
                the ones it does carry.
        """
        candidate = self.root / self.encoders_dir / cohort.lower()
        if not candidate.is_dir():
            available = sorted(
                path.name
                for path in (self.root / self.encoders_dir).glob("*")
                if path.is_dir()
            )
            raise FileNotFoundError(
                f"no archived encoder for {cohort!r} in {self.root}; "
                f"available: {available or 'none'}"
            )
        return candidate

    def read_pam50(self) -> list[str]:
        """Read the label-defining panel, or return an empty list if absent.

        Returns:
            Gene symbols, one per line of the panel file.
        """
        path = self.root / self.pam50_genes_file
        if not path.exists():
            return []
        return [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]


def rebase_config(
    config: ExperimentConfig,
    layout: BundleLayout,
    results_dir: Path | str,
) -> ExperimentConfig:
    """Rewrite a configuration's paths onto a cohort bundle.

    Only locations change. Every setting that decides what a run *computes* —
    the cohort composition, the fold count, the gene budget, the architecture —
    is left exactly as the configuration file states it, so a rebased run and a
    local one differ in where they read and nowhere else.

    Args:
        config: Configuration as loaded from ``configs/``.
        layout: Where the bundle's files sit.
        results_dir: Root to write results under.

    Returns:
        A new configuration; the original is not modified.
    """
    grn_path = layout.root / layout.grn_file
    embeddings = layout.root / layout.embeddings_file

    data = replace(
        config.data,
        raw_dir=str(layout.root),
        curated_genes_file=str(layout.root / layout.curated_genes_file),
        ploidy_file=(
            str(layout.root / Path(config.data.ploidy_file).name)
            if config.data.ploidy_file
            else None
        ),
    )
    folds = replace(config.folds, path=str(layout.root / layout.folds_file))
    graph = replace(
        config.graph,
        edges_file=str(layout.root / layout.edges_file),
        grn_file=str(grn_path) if grn_path.exists() else config.graph.grn_file,
    )
    model = config.model
    if config.model.pretrained_emb_file and embeddings.exists():
        model = replace(
            config.model,
            pretrained_emb_file=str(embeddings),
            pretrained_emb_symbol_file=None,
        )

    return replace(
        config,
        results_dir=str(results_dir),
        data=data,
        folds=folds,
        graph=graph,
        model=model,
    )


def provenance_record(extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Describe the machine and the moment a run happened.

    "It was slower/different that time" is only answerable if each result says
    what produced it: interpreter, library versions, device.

    Args:
        extra: Additional fields to record, such as the models requested.

    Returns:
        A JSON-serialisable record.
    """
    record: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import torch

        record["torch"] = torch.__version__
        record["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            record["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:  # pragma: no cover - torch is a hard dependency
        record["torch"] = None

    if extra:
        record.update(dict(extra))
    return record


def save_json(payload: Any, path: Path | str) -> Path:
    """Write a JSON document, creating parent directories.

    Args:
        payload: Anything :mod:`json` can serialise, with numpy scalars
            coerced to their Python equivalents.
        path: Destination file.

    Returns:
        The path written.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, default=_coerce) + "\n", encoding="utf-8"
    )
    return destination


def _coerce(value: Any) -> Any:
    """Render values json cannot handle natively.

    Args:
        value: The object :mod:`json` refused.

    Returns:
        A serialisable stand-in.
    """
    if isinstance(value, Path):
        return str(value)
    # An ndarray has .item() too, but it raises on anything but a single
    # element, so shape is what decides here rather than the method's presence.
    if hasattr(value, "tolist") and getattr(value, "ndim", 0) > 0:
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return str(value)
