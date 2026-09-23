"""Backend-neutral Kev-style weighting policy for the logical objective."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ObjectivePolicy:
    """Question losses average within a variant, then variants average equally."""

    question_reduction: Literal["mean"] = "mean"
    variant_reduction: Literal["mean"] = "mean"
    target_interpretation: Literal["categorical_cross_entropy"] = "categorical_cross_entropy"


KEV_OBJECTIVE_POLICY = ObjectivePolicy()
