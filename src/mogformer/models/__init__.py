"""The supervised classifier and the self-supervised encoder."""

from __future__ import annotations

from mogformer.models.classifier import MultiOmicsGraphClassifier
from mogformer.models.ssl import FUSION_TYPES, MOGFormerSSL

__all__ = ["FUSION_TYPES", "MOGFormerSSL", "MultiOmicsGraphClassifier"]
