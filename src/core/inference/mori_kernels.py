from dataclasses import dataclass
from typing import Literal
import numpy as np
from core.coarse_graining.filter import BlockCutoffFilter
from core.data.base import Field
from core.inference.base import InferenceMethod
from core.operators.base import Identity, OperatorBasis
from core.operators.mori_zwanzig import kept_indices, kept_modes
from core.rgflow.couplings import CouplingsForOperators

Origins = Literal["initial", "all"]


@dataclass
class MemoryKernels:
    """Kernels of the generalized Langevin equation at one filter level, on the coarse grid of that level.

    K = n_wavevectors if homogeneous, else = 1
    
    M = n_operators * n_components if homogeneous, else = n_wavevectors * n_operators * n_components"""

    correlations: np.ndarray  # (n_lags + 1, K, M, M) C(n) = <g(n) g(0)^H>
    kernels: np.ndarray  # (n_lags, K, M, M) Omega^(l): l = 0 the Markov term, l >= 1 the memory
    basis: OperatorBasis  # the operators O_i(u) whose modes are the observables g
    wavevectors: np.ndarray  # (n_wavevectors, n_space) the kept wavevectors, in the order of the modes
    homogeneous: bool
    scale: tuple[float, ...]  # grid spacing of the field
    dt: float  # time between snapshots, the step of the equation
    level: int = 0
    block_size: int = 1  # total coarse-graining factor relative to the original grid
    k_max: tuple[float, ...] | None = None

    @property
    def n_lags(self) -> int:
        return len(self.kernels)

    def predict(self, g: np.ndarray) -> np.ndarray:
        """sum_l Omega^(l) g(n-l), the step without the noise, from g(0..n) (batch, n + 1, K, M) -> (batch, K, M)."""
        n = g.shape[1] - 1
        if n >= self.n_lags:
            raise ValueError(f"{n + 1} states g(0..{n}) need {n + 1} kernels, only {self.n_lags} were fitted.")
        # g reversed in time holds g(n - l) at index l
        return np.einsum("lkij,blkj->bki", self.kernels[:n + 1], g[:, ::-1])

    def noise(self, g: np.ndarray) -> np.ndarray:
        """W(n) = g(n+1) - sum_l Omega^(l) g(n-l) along trajectories g (batch, time, K, M) -> (batch, n_steps, K, M), n_steps = min(time - 1, n_lags)."""
        n_steps = min(g.shape[1] - 1, self.n_lags)
        return np.stack([g[:, n + 1] - self.predict(g[:, :n + 1]) for n in range(n_steps)], axis=1)

    def rollout(self, g0: np.ndarray, n_steps: int) -> np.ndarray:
        """g(0..n_steps) without the noise from g(0) (batch, K, M) -> (batch, n_steps + 1, K, M)."""
        g = g0[:, None]
        for _ in range(n_steps):
            g = np.concatenate([g, self.predict(g)[:, None]], axis=1)
        return g

    def couplings(self, lag: int = 0) -> CouplingsForOperators:
        """Omega^(lag) grouped by the input operator O_i: for every O_i, the coefficients of its modes in the output modes, (K, M, n_components) if homogeneous, else (M, n_wavevectors, n_components)."""
        kernel = self.kernels[lag]  # (K, M, M)
        n_ops = len(self.basis)
        if self.homogeneous:
            split = kernel.reshape(*kernel.shape[:2], n_ops, -1)  # (K, M, n_ops, n_components)
        else:
            split = kernel[0].reshape(kernel.shape[1], len(self.wavevectors), n_ops, -1)  # (M, n_wavevectors, n_ops, n_components)
        return CouplingsForOperators(self.basis, [split[:, :, i] for i in range(n_ops)], scale=self.scale)

    def markov(self) -> CouplingsForOperators:
        return self.couplings(0)


def observables(field: Field, basis: OperatorBasis | None = None, homogeneous: bool = True) -> np.ndarray:
    """The kept Fourier modes of O_i(field) as observables g: (batch, time, K, M), see MemoryKernels."""
    features = field if basis is None else basis.stack(field)
    modes = kept_modes(BlockCutoffFilter(block_size=1), features)
    if not homogeneous:
        return modes[:, :, None, :]
    M = int(np.prod(features.component_shape))
    return modes.reshape(*modes.shape[:2], -1, M)


def two_time_correlations(g: np.ndarray, n_lags: int | None = None, origins: Origins = "initial") -> np.ndarray:
    """C(n) = <g(n0 + n) g(n0)^H> for n = 0..n_lags, the mean over the batch and the time origins n0 (0 if origins = "initial, else every n0 with n0 + n_lags inside the trajectories): (n_lags + 1, K, M, M) from g (batch, time, K, M)."""
    n_time: int = g.shape[1]
    n_lags = n_time - 1 if n_lags is None else n_lags
    if not 1 <= n_lags < n_time:
        raise ValueError(f"n_lags must be between 1 and {n_time - 1} for {n_time} time steps, got {n_lags}.")
    if origins == "initial":
        starts = [0]
    elif origins == "all":
        starts = range(n_time - n_lags)
    else:
        raise ValueError(f"origins must be 'initial' or 'all', got {origins!r}.")
    per_origin = [np.einsum("btki,bkj->tkij", g[:, s:s + n_lags + 1], g[:, s].conj()) for s in starts]
    return np.stack(per_origin).mean(axis=0) / len(g)


def memory_kernels(C: np.ndarray, rcond: float = 1e-10) -> np.ndarray:
    """Omega^(0..n_lags-1) from the correlations C(0..n_lags) (n_lags + 1, K, M, M) by the recursion C(n+1) = sum_{l=0..n} Omega^(l) C(n-l)."""
    C0_inv = np.linalg.pinv(C[0], rcond=rcond, hermitian=True)
    omega = []
    for n in range(len(C) - 1):
        omega.append((C[n + 1] - sum(omega[l] @ C[n - l] for l in range(n))) @ C0_inv)
    return np.stack(omega)


def mori_kernels_level(
    u: Field,
    basis: OperatorBasis | None = None,
    homogeneous: bool = True,
    n_lags: int | None = None,
    origins: Origins = "initial",
    rcond: float = 1e-10,
    level: int = 0,
    block_size: int = 1,
    k_max: tuple[float, ...] | None = None,
) -> MemoryKernels:
    """The kernels of trajectories u (batch, time, ...) resolved on their own grid, with equally spaced snapshots."""
    dt = _check_trajectories(u)
    operators = OperatorBasis([Identity()]) if basis is None else basis
    g = observables(u, basis, homogeneous)
    C = two_time_correlations(g, n_lags, origins)
    features = u if basis is None else basis.stack(u)
    kept = [k[idx] for k, idx in zip(features.wavenumbers(), kept_indices(BlockCutoffFilter(block_size=1), features))]
    return MemoryKernels(
        correlations=C,
        kernels=memory_kernels(C, rcond),
        basis=operators,
        wavevectors=np.stack(np.meshgrid(*kept, indexing="ij"), axis=-1).reshape(-1, len(kept)),
        homogeneous=homogeneous,
        scale=u.scale,
        dt=dt,
        level=level,
        block_size=block_size,
        k_max=tuple(np.pi / dx for dx in u.scale) if k_max is None else k_max,
    )


def _check_trajectories(field: Field) -> float:
    """Return the time difference between snapshots after checking that field is batched with at least 2 equally spaced ones."""
    if not field.batched or field.time is None:
        raise ValueError("the memory kernels need a batched field with a time axis.")
    if len(field.time) < 2:
        raise ValueError(f"not enough time steps ({len(field.time)}): C(1) needs at least 2.")
    steps = np.diff(field.time)
    if not np.allclose(steps, steps[0]):
        raise ValueError("the snapshots must be equally spaced in time: the kernels are per step.")
    return float(steps[0])


class MoriKernels(InferenceMethod):
    """Infer Markov and memory kernels of the generalized Langevin equation from two-time correlations of batched trajectories (batch, time, ...) resolved on their own grid."""

    def __init__(
        self, homogeneous: bool = True, n_lags: int | None = None, origins: Origins = "initial", rcond: float = 1e-10
    ):
        self.homogeneous = homogeneous
        self.n_lags = n_lags
        self.origins: Origins = origins
        self.rcond = rcond

    def fit(self, field: Field, basis: OperatorBasis) -> CouplingsForOperators:
        return self.fit_kernels(field, basis).markov()

    def fit_kernels(self, field: Field, basis: OperatorBasis) -> MemoryKernels:
        return mori_kernels_level(field, basis, self.homogeneous, self.n_lags, self.origins, self.rcond)
