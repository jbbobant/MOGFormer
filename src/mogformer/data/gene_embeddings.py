"""Pretrained gene embeddings, aligned to a fold's selected genes.

The FROGS/ARCHS4 table is a fixed 256-dimensional vector per gene symbol,
learned from bulk expression atlases outside this cohort. It is a *prior over
genes*, not over patients, so using it introduces no leakage: the same vector
is attached to a gene whether that gene's patient sits in the training or the
held-out fold.

The table ships as a headerless matrix plus a row-aligned symbol list, which is
why loading it needs two paths rather than one.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_gene_embedding_table(
    matrix_file: str | Path, symbol_file: str | Path
) -> pd.DataFrame:
    """Read the pretrained embedding matrix indexed by gene symbol.

    Args:
        matrix_file: Headerless CSV of shape ``(n_rows, d_embedding)``.
        symbol_file: One gene symbol per line, row-aligned to ``matrix_file``.

    Returns:
        Frame indexed by gene symbol. Duplicate symbols are collapsed by mean,
        which is how the source table's isoform rows are meant to combine.

    Raises:
        ValueError: If the matrix and the symbol list disagree in length.
    """
    matrix = pd.read_csv(matrix_file, header=None).to_numpy(dtype=np.float32)
    symbols = [
        line.strip()
        for line in Path(symbol_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(symbols) != matrix.shape[0]:
        raise ValueError(
            f"{len(symbols)} symbols but {matrix.shape[0]} embedding rows; "
            "the symbol list must be row-aligned to the matrix"
        )
    frame = pd.DataFrame(matrix, index=pd.Index(symbols, name="gene"))
    if frame.index.duplicated().any():
        frame = frame.groupby(level=0).mean()
    logger.info(
        "gene embeddings: %d symbols x %d dimensions", len(frame), frame.shape[1]
    )
    return frame


def align_gene_embeddings(
    table: pd.DataFrame, genes: Sequence[str]
) -> tuple[np.ndarray, list[str]]:
    """Select embedding rows for a fold's genes, in the fold's own order.

    Row ``i`` of the result must correspond to ``genes[i]``, because every
    downstream shape contract keys off gene position. Genes the table does not
    cover therefore keep a row — filled with zeros — rather than being dropped.

    Args:
        table: Embeddings indexed by gene symbol.
        genes: Gene symbols in the order the model expects them.

    Returns:
        Tuple of the ``(len(genes), d_embedding)`` matrix and the symbols that
        were actually covered by the table.
    """
    covered = [gene for gene in genes if gene in table.index]
    matrix = np.zeros((len(genes), table.shape[1]), dtype=np.float32)
    if covered:
        positions = [i for i, gene in enumerate(genes) if gene in table.index]
        matrix[positions] = table.loc[covered].to_numpy(dtype=np.float32)
    if len(covered) < len(genes):
        logger.info(
            "gene embeddings cover %d of %d selected genes; the rest are zeroed",
            len(covered),
            len(genes),
        )
    return matrix, covered


def save_gene_embedding_table(table: pd.DataFrame, path: str | Path) -> Path:
    """Write an embedding table as a single indexed archive.

    The source pair is 68 MB of text for a table that compresses to a fraction
    of that, and loads in a fraction of the time.

    Args:
        table: Embeddings indexed by gene symbol.
        path: Destination ``.npz`` file. Parent directories are created.

    Returns:
        The path written.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        values=table.to_numpy(dtype=np.float32),
        genes=np.asarray(list(table.index), dtype=object),
    )
    return destination


def load_gene_embedding_bundle(path: str | Path) -> pd.DataFrame:
    """Read a table written by :func:`save_gene_embedding_table`.

    Args:
        path: Archive to read.

    Returns:
        Frame indexed by gene symbol.
    """
    with np.load(Path(path), allow_pickle=True) as archive:
        return pd.DataFrame(
            archive["values"],
            index=pd.Index([str(gene) for gene in archive["genes"]], name="gene"),
        )
