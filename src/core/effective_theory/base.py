from dataclasses import dataclass
import numpy as np
from core.coarse_graining.filter import BlockCutoffFilter
from core.data.base import Field
from core.operators.base import OperatorBasis
from core.operators.mori_zwanzig import apply_coefficients, kept_indices, kept_modes
from core.rgflow.couplings import CouplingsForOperators


def _number(x) -> str:
    x = complex(x)
    return f"{x.real:+.6f}" if abs(x.imag) < 1e-12 else f"{x:+.6f}"


def _format(value) -> str:
    """A scalar coupling as its value; an array coupling as mean±std over its entries, then its shape."""
    value = np.asarray(value)
    if value.ndim == 0:
        return _number(value.item())
    return f"{_number(value.mean())}±{value.std():.6f} {list(value.shape)}"


def _check_names(basis: OperatorBasis, couplings: CouplingsForOperators, what: str):
    if basis.names != couplings.names:
        raise ValueError(
            f"There is a mismatch between the operators and {what}! "
            f"Got operators {basis.names} and {what} {couplings.names}"
        )


@dataclass
class EffectiveTheory:
    basis: OperatorBasis
    couplings: CouplingsForOperators

    def __post_init__(self):
        _check_names(self.basis, self.couplings, "couplings")

    @property
    def scale(self):
        return self.couplings.scale

    def evaluate(self, field: Field) -> Field:
        """sum_i K_i O_i(field)"""
        terms = self.basis.evaluate(field)
        values = [np.asarray(self.couplings[name]) for name in self.basis.names]
        if any(v.ndim for v in values):
            raise ValueError("evaluate needs one coupling per operator. Got couplings with shapes: " + ", ".join(f"{name}: {v.shape}" for name, v in zip(self.basis.names, values)))
        return field.replace(values=sum(v * term.values for v, term in zip(values, terms)))

    def describe(self):
        terms = []

        for name, value in self.couplings.values.items():
            terms.append(
                f"{_format(value)} * {name}"
            )

        return "H_eff = " + " ".join(terms)


@dataclass
class EffectiveDynamics(EffectiveTheory):
    """
    Stochastic coarse dynamics at one scale, from a Mori-Zwanzig fit on trajectories:

        u[t+1] - u[t] = sum_i (A_i + gamma(t) B_i) O_i(u[t]) + noise[t],    E|noise_k[t]|^2 = noise_var[t, k]

    """

    memory: CouplingsForOperators
    ratio: CouplingsForOperators  # "gamma": (n_time,), with its stderr
    noise: CouplingsForOperators  # "noise_var": (n_time, *space, *components), per Fourier mode
    time: np.ndarray  # (n_time,) time steps of ratio and noise
    wavevectors: np.ndarray  # (n_wavevectors, n_space) the kept wavevectors of the couplings
    homogeneous: bool

    def __post_init__(self):
        _check_names(self.basis, self.couplings, "markov couplings")
        _check_names(self.basis, self.memory, "memory couplings")
        if self.ratio.names != ["gamma"] or self.noise.names != ["noise_var"]:
            raise ValueError(f"expected couplings 'gamma' and 'noise_var', got {self.ratio.names} and {self.noise.names}")

    @property
    def markov(self) -> CouplingsForOperators:
        return self.couplings

    def drift(self, t: int) -> CouplingsForOperators:
        """The couplings A_i + gamma(t) B_i of the effective drift at time step index t."""
        gamma = self.ratio["gamma"][t]
        values = [self.markov[name] + gamma * self.memory[name] for name in self.basis.names]
        return CouplingsForOperators(self.basis, values, scale=self.markov.scale)

    def evaluate(self, field: Field, t: int | None = None, memory: bool = False) -> Field:
        if field.time is not None:
            if t is not None:
                raise ValueError("field has a time axis, so gamma is taken at field.time: use memory=True, not t.")
            if not memory:
                return self._apply(self.couplings, field)
            gamma = self.ratio["gamma"][self._time_indices(field.time)]
            shape = [1] * field.values.ndim
            shape[field.n_batch] = -1  # field.n_batch corresponds to the index of the time axis
            gamma = gamma.reshape(shape)
            markov = self._apply(self.markov, field, fourier=True)
            drift = markov.replace(values=markov.values + gamma * self._apply(self.memory, field, fourier=True).values)
            return self._to_domain(drift, field)
        if memory and t is None:
            raise ValueError("field has no time axis: give the time step index t of the memory.")
        return self._apply(self.couplings if t is None else self.drift(t), field)

    def _time_indices(self, times: np.ndarray) -> np.ndarray:
        """Indices into self.time of times, which must all be time steps of the dynamics."""
        idx = np.clip(np.searchsorted(self.time, times), 0, len(self.time) - 1)
        if not np.array_equal(self.time[idx], times):
            raise ValueError(f"the time steps {times} of field must be among the time steps {self.time} of gamma.")
        return idx

    def _to_domain(self, drift: Field, field: Field) -> Field:
        if field.domain == "real":
            return drift.to_real(real=not np.iscomplexobj(field.values))
        return drift

    def _apply(self, couplings: CouplingsForOperators, field: Field, fourier: bool = False) -> Field:
        """sum_i K_i O_i(field) for the couplings K_i, in the domain of field or, with fourier=True, in Fourier space."""
        features = self.basis.stack(field)
        resolved = BlockCutoffFilter(block_size=1)
        modes = kept_modes(resolved, features)
        kept = kept_indices(resolved, features)
        if int(np.prod([len(idx) for idx in kept])) != len(self.wavevectors):
            raise ValueError(f"field must be on the grid of the dynamics, with {len(self.wavevectors)} wavevectors.")
        stacked = np.stack([couplings[name] for name in self.basis.names], axis=1)  # (n_wavevectors, n_ops, n_comp, ...)
        if self.homogeneous:
            coeffs = stacked.reshape(stacked.shape[0], -1, *stacked.shape[3:])
        else:
            coeffs = stacked.reshape(-1, *stacked.shape[3:])
        drift = field.to_fourier().replace(
            values=apply_coefficients(modes, coeffs, kept, field.spatial_shape, self.homogeneous)
        )
        return drift if fourier else self._to_domain(drift, field)

    def describe(self):
        terms = [
            f"({_format(self.markov[name])} + gamma(t) {_format(self.memory[name])}) * {name}" for name in self.basis.names
        ]
        return "du = " + " + ".join(terms) + " + noise"
