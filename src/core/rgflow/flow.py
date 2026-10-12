import warnings
from dataclasses import dataclass, field
from typing import Literal
from core.effective_theory.base import EffectiveDynamics, EffectiveTheory
from core.operators.base import Operator, OperatorBasis
from core.rgflow.couplings import CouplingsForOperators

Part = Literal["couplings", "memory", "ratio", "noise"]


class _NamedOperator(Operator):
    """Placeholder for an operator known only by its name, from a flow point given as bare couplings."""

    def __init__(self, name: str):
        self.name = name

    def evaluate(self, field):
        raise NotImplementedError(
            f"operator {self.name!r} is only known by name: build the flow from EffectiveTheory points to evaluate it."
        )


def _as_theory(point: object) -> EffectiveTheory:
    if isinstance(point, EffectiveTheory):
        return point
    if isinstance(point, CouplingsForOperators):
        warnings.warn(
            "RGFlow points should be EffectiveTheory instances: wrap the couplings as "
            "EffectiveTheory(basis, couplings). Bare CouplingsForOperators get a basis known only by name.",
            DeprecationWarning,
            stacklevel=3,
        )
        return EffectiveTheory(OperatorBasis([_NamedOperator(name) for name in point.names]), point)
    raise TypeError(f"an RGFlow point must be an EffectiveTheory, got {type(point).__name__}.")


@dataclass
class RGFlow:
    points: list[EffectiveTheory] = field(default_factory=list)

    def __post_init__(self):
        self.points = [_as_theory(p) for p in self.points]

    def add(self, theory: EffectiveTheory):
        self.points.append(_as_theory(theory))

    @property
    def scales(self):
        return [p.scale for p in self.points]

    def values(self, name: str, part: Part = "couplings"):
        """The coupling name of every point, from the given set of couplings: "couplings" (the Markov couplings of EffectiveDynamics), or "memory", "ratio" ("gamma") and "noise" ("noise_var"), which only EffectiveDynamics has."""
        if part not in ("couplings", "memory", "ratio", "noise"):
            raise ValueError(f"part must be 'couplings', 'memory', 'ratio' or 'noise', got {part!r}.")
        if part != "couplings" and not all(isinstance(p, EffectiveDynamics) for p in self.points):
            raise ValueError(f"part {part!r} needs every point to be an EffectiveDynamics.")
        return [getattr(p, part)[name] for p in self.points]

    def __len__(self):
        return len(self.points)
