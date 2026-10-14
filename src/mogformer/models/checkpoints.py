"""Load encoders saved under an earlier spelling of their parameter names.

The archived self-supervised checkpoints store several parameters under short
names — ``b_grn_act`` for ``b_grn_activation``, ``ffn_g`` for ``ffn_global``.
Left unhandled, a name mismatch silently turns every one of those tensors into
a freshly initialised one; PyTorch reports it as a list of missing keys that is
easy to skim past.

So the rename is written down rather than discovered, and loading is strict by
default: a checkpoint that does not map cleanly onto the current model raises
instead of quietly producing a half-random encoder.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)

#: Parameter prefixes as stored in the archived checkpoints, mapped onto
#: their current spelling. Applied longest-first so a shorter prefix cannot
#: shadow a longer one.
LEGACY_PREFIX_RENAMES: Mapping[str, str] = {
    "decoder.ffn_g.": "decoder.ffn_global.",
    "decoder.ffn_l.": "decoder.ffn_local.",
    "decoder.lin_g.": "decoder.lin_global.",
    "decoder.lin_l.": "decoder.lin_local.",
}

#: Parameter names renamed outright, matched on the trailing segment so they
#: apply at any depth of the module tree.
LEGACY_SUFFIX_RENAMES: Mapping[str, str] = {
    "attn.b_grn_act": "attn.b_grn_activation",
    "attn.b_grn_repr": "attn.b_grn_repression",
    "masker.mod_w": "masker.modality_weights",
}


def rename_legacy_keys(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Translate an archived state dict onto the current parameter names.

    Args:
        state_dict: Parameters as stored in the checkpoint.

    Returns:
        The same tensors under the names the current model uses.
    """
    renamed: dict[str, Any] = {}
    prefixes = sorted(LEGACY_PREFIX_RENAMES, key=len, reverse=True)

    for key, value in state_dict.items():
        new_key = key
        for old in prefixes:
            if new_key.startswith(old):
                new_key = LEGACY_PREFIX_RENAMES[old] + new_key[len(old) :]
                break
        for old, new in LEGACY_SUFFIX_RENAMES.items():
            if new_key.endswith(old):
                new_key = new_key[: -len(old)] + new
                break
        renamed[new_key] = value

    changed = sum(1 for old, new in zip(state_dict, renamed, strict=True) if old != new)
    if changed:
        logger.info("renamed %d legacy parameter(s) while loading", changed)
    return renamed


def load_encoder_state(
    model: nn.Module, state_dict: Mapping[str, Any], strict: bool = True
) -> nn.Module:
    """Load parameters into a model, translating legacy names first.

    Args:
        model: Model to load into.
        state_dict: Parameters, in either spelling.
        strict: Refuse a checkpoint that does not map cleanly. Turning this off
            is only reasonable when the mismatch has been inspected.

    Returns:
        The model, loaded.

    Raises:
        ValueError: Under ``strict``, if any parameter is missing or unused,
            naming them so the rename table can be extended.
    """
    renamed = rename_legacy_keys(state_dict)
    missing, unexpected = model.load_state_dict(renamed, strict=False)

    if strict and (missing or unexpected):
        raise ValueError(
            "checkpoint does not match the model.\n"
            f"  missing ({len(missing)}): {sorted(missing)[:10]}\n"
            f"  unused  ({len(unexpected)}): {sorted(unexpected)[:10]}\n"
            "Extend LEGACY_PREFIX_RENAMES/LEGACY_SUFFIX_RENAMES if this is a "
            "rename, or rebuild the model with the checkpoint's own settings."
        )
    if missing or unexpected:
        logger.warning(
            "loaded with %d missing and %d unused parameter(s)",
            len(missing),
            len(unexpected),
        )
    return model


def read_checkpoint(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read an archived encoder file into its parameters and its settings.

    Args:
        path: A ``.pt`` file holding ``state_dict`` and ``config``.

    Returns:
        Tuple of the parameters and the settings the run was trained under.

    Raises:
        ValueError: If the file is not in that shape.
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(
            f"{path} does not hold a checkpoint with a 'state_dict' entry; "
            f"found {type(payload).__name__}"
        )
    return dict(payload["state_dict"]), dict(payload.get("config", {}))
