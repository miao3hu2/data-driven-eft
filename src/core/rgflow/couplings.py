from collections.abc import Sequence
from typing import cast
import numpy as np
from numpy.typing import ArrayLike
from core.operators.base import OperatorBasis


class CouplingsForOperators:

    def __init__(
        self,
        basis: OperatorBasis | Sequence[str],
        values: ArrayLike | Sequence[ArrayLike],
        scale: float | Sequence[float],
        stderr: dict[str, ArrayLike] | None = None,
    ):
        names = basis.names if isinstance(basis, OperatorBasis) else list(basis)
        # a list holds one coupling per operator, possibly of different shapes; an array holds one per leading entry
        if isinstance(values, (list, tuple)):
            entries = [np.asarray(v) for v in values]
        else:
            entries = list(np.atleast_1d(cast(ArrayLike, values)))

        if len(entries) != len(names):
            raise ValueError(f"expected {len(names)} values, got {len(entries)}")
        if len(set(names)) != len(names):
            raise ValueError("basis contains duplicate operator names")
        if stderr is not None and not set(stderr) <= set(names):
            raise ValueError(f"stderr names {sorted(stderr)} are not all operators of {names}")
        self.values = dict(zip(names, entries))
        self.stderr = None if stderr is None else {name: np.asarray(s) for name, s in stderr.items()}
        self.scale = scale

    @property
    def names(self) -> list[str]:
        return list(self.values)

    def __getitem__(self, name: str):
        return self.values[name]

    def get(self, name: str, default=None):
        return self.values.get(name, default)

    def as_array(self) -> np.ndarray:
        """The couplings stacked in operator order: (n_operators, *coupling shape)."""
        values = np.array(list(self.values.values()))
        return values if np.iscomplexobj(values) else values.astype(float)
