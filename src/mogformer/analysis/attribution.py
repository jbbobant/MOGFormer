"""Attributing a score read off the frozen embedding back to the inputs.

Any linear read-out of the patient embedding (a Cox risk score, a stratum
classifier) is a function of the encoder's inputs. Integrated gradients
(Sundararajan, Taly & Yan 2017) splits that score, patient by patient, over
the input values (gene by omics): it integrates the gradient along the
straight path from a baseline to the patient's own inputs. Its attributions
sum to the score's difference from the baseline (completeness), so a gene's
share is a share of the actual prediction, not of an arbitrary saliency scale.

The baseline is the all-zero input, which in the encoder's standardised units
is the average patient. A gene's attribution therefore says how much the
patient's departure from average on that gene moved their score.

The encoder's value tokenizer is steep near zero: most of a score's change
happens in the first percent of the path. Evenly spaced points would miss it
and break completeness, so the path is sampled on a logarithmic grid from
``1e-4`` to 1 (plus the baseline itself) and integrated by the trapezoid rule.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)


def integrated_gradients(
    model: nn.Module,
    inputs: torch.Tensor,
    direction: torch.Tensor,
    graph_pe: torch.Tensor,
    spd: torch.Tensor,
    grn: torch.Tensor | None,
    device: torch.device,
    steps: int = 32,
    batch_size: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Attribute the score ``embedding · direction`` to every input value.

    Args:
        model: The frozen encoder; its patient embedding is ``out["c"]``.
        inputs: Standardised inputs, shape ``(n_patients, n_genes, 3)``.
        direction: Read-out weights over the embedding, shape ``(width,)``.
        graph_pe: Positional encodings.
        spd: Shortest-path matrix.
        grn: Signed regulatory adjacency, or None.
        device: Where to compute.
        steps: Points on the integration path (log-spaced, trapezoid rule).
        batch_size: Patients per pass; each pass evaluates
            ``batch_size * steps`` inputs.

    Returns:
        Attributions of shape ``(n_patients, n_genes, 3)``, and the
        completeness gap per patient: the score's difference from the
        baseline minus the attributions' sum, which should be near zero.
    """
    model = model.to(device).eval()
    direction = direction.to(device, torch.float32)
    pe, spd, grn = (
        graph_pe.to(device),
        spd.to(device),
        None if grn is None else grn.to(device),
    )
    grid = np.concatenate([[0.0], np.logspace(-4, 0, steps - 1)])
    widths = np.zeros(steps)
    widths[1:] += np.diff(grid) / 2
    widths[:-1] += np.diff(grid) / 2
    alphas = torch.as_tensor(grid, device=device, dtype=torch.float32)
    weights = torch.as_tensor(widths, device=device, dtype=torch.float32)

    def score(values: torch.Tensor) -> torch.Tensor:
        out = model(
            values[..., 0], values[..., 1], values[..., 2], pe, spd, grn, mask=False
        )
        return out["c"] @ direction

    attributions, gaps = [], []
    for start in range(0, inputs.shape[0], batch_size):
        x = inputs[start : start + batch_size].to(device, torch.float32)
        path = (alphas.view(-1, 1, 1, 1) * x.unsqueeze(0)).reshape(-1, *x.shape[1:])
        path.requires_grad_(True)
        (gradient,) = torch.autograd.grad(score(path).sum(), path)
        mean_gradient = (
            gradient.reshape(steps, *x.shape) * weights.view(-1, 1, 1, 1)
        ).sum(dim=0)
        attribution = (x * mean_gradient).detach()
        with torch.no_grad():
            gap = score(x) - score(torch.zeros_like(x)) - attribution.sum(dim=(1, 2))
        attributions.append(attribution.cpu())
        gaps.append(gap.cpu())
    result = torch.cat(attributions).numpy()
    completeness_gap = torch.cat(gaps).numpy()
    logger.info(
        "integrated gradients: %d patients, median |completeness gap| %.4f "
        "(median |score| %.4f)",
        len(result),
        float(np.median(np.abs(completeness_gap))),
        float(np.median(np.abs(result.sum(axis=(1, 2)) + completeness_gap))),
    )
    return result, completeness_gap
