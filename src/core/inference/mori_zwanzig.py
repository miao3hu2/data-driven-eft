import warnings
from dataclasses import dataclass
import numpy as np
from core.coarse_graining.filter import BlockCutoffFilter
from core.data.base import Field
from core.effective_theory.base import EffectiveDynamics
from core.inference.base import InferenceMethod
from core.operators.base import Identity, OperatorBasis
from core.operators.mori_zwanzig import Projection
from core.rgflow.couplings import CouplingsForOperators


@dataclass
class MoriZwanzigData:
    """Residual and memory data at one filter level, on the coarse grid of that level."""

    level: int
    block_size: int  # total coarse-graining factor relative to the original grid
    scale: tuple[float, ...]  # coarse grid spacing (L / grid size)
    k_max: tuple[float, ...]  # cutoff pi / (block_size * dx) along each spatial axis: the modes with |k| < k_max are kept
    residual: Field  # u_n[:, t+1] - u_n[:, t] - P L u_n[:, t]: (batch, n_time - 2, *coarse space, *components)
    memory: Field  # P L Q L u_n[:, t]: same layout as residual
    basis: OperatorBasis  # the operators O_i(u) that P projects on
    markov: CouplingsForOperators  # couplings A_i of P L u = sum_i A_i O_i(u), see EffectiveDynamics
    memory_couplings: CouplingsForOperators  # couplings B_i of P L Q L u = sum_i B_i O_i(u)
    wavevectors: np.ndarray  # (n_wavevectors, n_space) the kept wavevectors of the couplings
    homogeneous: bool

    def ratio(self) -> tuple[np.ndarray, np.ndarray]:
        """
        gamma(t) = argmin_gamma sum_b |Q L u[b, t] - gamma P L Q L u[b, t]|^2, summed over the batch, the grid and the
        components, and its standard error from the spread over the batch members: (n_time - 2,) each.

        gamma is nan where the memory vanishes on every batch member.
        """
        num, den = self.ratio_sums()
        n_batch = num.shape[0]
        total = den.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            gamma = num.sum(axis=0) / total
            # sandwich variance with the batch members as independent clusters
            spread = num - gamma * den
            stderr = np.sqrt(n_batch / max(n_batch - 1, 1) * np.sum(np.abs(spread) ** 2, axis=0)) / total
        return gamma, stderr

    def ratio_sums(self) -> tuple[np.ndarray, np.ndarray]:
        """Per batch member and time step: sum conj(m) r and sum |m|^2 over the grid and components, with r the residual and m the memory: (batch, n_time - 2) each."""
        r, m = self.residual.values, self.memory.values
        axes = tuple(range(2, r.ndim))
        num = np.sum(m.conj() * r, axis=axes)
        den = np.sum(np.abs(m) ** 2, axis=axes)
        return num, den

    def noise(self) -> Field:
        """residual - gamma(t) memory: the noise of every batch member around the fitted center, same layout as residual."""
        gamma, _ = self.ratio()
        gamma = gamma.reshape(-1, *[1] * (self.memory.values.ndim - 2))
        return self.residual.replace(values=self.residual.values - gamma * self.memory.values)

    def noise_var(self) -> np.ndarray:
        """Mean |noise_k|^2 over the batch members per Fourier mode: (n_time - 2, *coarse space, *components)."""
        return np.mean(np.abs(self.noise().to_fourier().values) ** 2, axis=0)

    def dynamics(self) -> EffectiveDynamics:
        """The coarse stochastic dynamics u[t+1] - u[t] = sum_i (A_i + gamma(t) B_i) O_i(u[t]) + noise at this level."""
        gamma, stderr = self.ratio()
        assert self.residual.time is not None
        return EffectiveDynamics(
            basis=self.basis,
            couplings=self.markov,
            memory=self.memory_couplings,
            ratio=CouplingsForOperators(["gamma"], [gamma], scale=self.scale, stderr={"gamma": stderr}),
            noise=CouplingsForOperators(["noise_var"], [self.noise_var()], scale=self.scale),
            time=self.residual.time,
            wavevectors=self.wavevectors,
            homogeneous=self.homogeneous,
        )


def mori_zwanzig_level(
    u: Field,
    basis: OperatorBasis | None = None,
    homogeneous: bool = True,
    level: int = 0,
    block_size: int = 1,
    k_max: tuple[float, ...] | None = None,
) -> MoriZwanzigData:
    check_trajectories(u)
    P = _projection(BlockCutoffFilter(block_size=1), u, homogeneous, basis)
    operators = OperatorBasis([Identity()]) if basis is None else basis
    Lu = _increment(u)
    c_Lu = P.coefficients(Lu)
    QLu = Lu.replace(values=Lu.values - P.apply(Lu, c_Lu).values)
    LQLu = _increment(QLu)
    c_LQLu = P.coefficients(LQLu)
    return MoriZwanzigData(
        level=level,
        block_size=block_size,
        scale=u.scale,
        k_max=tuple(np.pi / dx for dx in u.scale) if k_max is None else k_max,
        residual=_first(QLu, _n_time(LQLu)),
        memory=P.apply(LQLu, c_LQLu),
        basis=operators,
        markov=_couplings(P, operators, c_Lu),
        memory_couplings=_couplings(P, operators, c_LQLu),
        wavevectors=P.wavevectors(),
        homogeneous=homogeneous,
    )


def check_trajectories(field: Field) -> None:
    if not field.batched or not field.has_time:
        raise ValueError("mori_zwanzig_data needs a batched field with a time axis.")
    if _n_time(field) < 3:
        raise ValueError(f"not enough time steps ({_n_time(field)}): L Q L u[t] needs u[t + 2], so at least 3.")


def _increment(field: Field) -> Field:
    """L g(t) = g(t+1) - g(t) along the time axis of a batched field."""
    assert field.time is not None
    return field.replace(values=np.diff(field.values, axis=1), time=field.time[:-1])


def _projection(resolved: BlockCutoffFilter, u: Field, homogeneous: bool, basis: OperatorBasis | None) -> Projection:
    projection = Projection(resolved, u, homogeneous=homogeneous, operators=basis)
    n_samples = u.shape[0]
    # unknowns per solve: all modes at once, or the components of one wavenumber
    n_unknowns = projection.gram.shape[-1]
    if n_samples <= n_unknowns:
        warnings.warn(
            f"{n_samples} samples for {n_unknowns} unknowns per solve on a {u.spatial_shape} grid: the Gram matrix is "
            "rank deficient, so P fits the samples exactly and the residual is not meaningful. Use more samples"
            + (" or more filter levels." if not homogeneous else "."),
            stacklevel=3,
        )
    return projection


def _couplings(P: Projection, basis: OperatorBasis, coeffs: np.ndarray) -> CouplingsForOperators:
    """The coefficients coeffs = P.coefficients(g) of P g grouped by the operators O_i of basis that P projects on: for every O_i (n_wavevectors, n_components of O_i, *rest) with rest as in Projection.coefficients, the wavevectors those of P.wavevectors()."""
    rest = coeffs.shape[2:] if P.homogeneous else coeffs.shape[1:]
    split = coeffs.reshape(len(P.wavevectors()), len(basis), -1, *rest)
    return CouplingsForOperators(basis.names, [split[:, i] for i in range(len(basis))], scale=P.features.scale)


def _n_time(field: Field) -> int:
    assert field.time is not None
    return len(field.time)


def _first(field: Field, n: int) -> Field:
    assert field.time is not None
    return field.replace(values=field.values[:, :n], time=field.time[:n])


class MoriZwanzig(InferenceMethod):
    """Couplings of the coarse dynamics u[t+1] - u[t] = sum_i (A_i + gamma(t) B_i) O_i(u[t]) + noise, from batched trajectories (batch, time, ...) of a field resolved on its own grid. Fit returns the Markov couplings A_i; fit_dynamics everything."""

    def __init__(self, homogeneous: bool = True):
        self.homogeneous = homogeneous

    def fit(self, field: Field, basis: OperatorBasis) -> CouplingsForOperators:
        return self.fit_dynamics(field, basis).markov

    def fit_dynamics(self, field: Field, basis: OperatorBasis) -> EffectiveDynamics:
        return mori_zwanzig_level(field, basis, self.homogeneous).dynamics()


class MemoryRatio(InferenceMethod):
    """The ratio gamma(t) in Q L u[t] = gamma(t) P L Q L u[t] + noise, with its stderr, as the coupling "gamma"."""

    def __init__(self, homogeneous: bool = True):
        self.homogeneous = homogeneous

    def fit(self, field: Field, basis: OperatorBasis) -> CouplingsForOperators:
        return MoriZwanzig(self.homogeneous).fit_dynamics(field, basis).ratio
