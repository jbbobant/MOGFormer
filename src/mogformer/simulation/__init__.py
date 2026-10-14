"""Synthetic cohorts with a known causal structure.

Used to score methods whose real-data answers have no ground truth — above all
the in-silico perturbation probes.
"""

from __future__ import annotations

from mogformer.simulation.scm import (
    GeneRegulatorySCM,
    SimulatedCohort,
    SimulationSettings,
)

__all__ = ["GeneRegulatorySCM", "SimulatedCohort", "SimulationSettings"]
