from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from core.coarse_graining.filter import BlockCutoffFilter
from core.data.base import Field
from core.data.burgers import Burgers
from core.inference.mori_kernels import MemoryKernels, MoriKernels, covariance_inverse, observables
from core.operators.base import Identity, Operator, OperatorBasis
from core.operators.symbolic import SymbolicOperator


@dataclass(frozen=True)
class Levels:
    fit: tuple[int, ...] = (1, 2, 3, 4)  # levels the model is fitted at; level n has block size 2^n
    predict: int = 5  # the level the flow is extrapolated to and scored at
    n_extrapolation_levels: int = 3  # number of fit levels used to extrapolate to the prediction scale


class Dynamics(Protocol):
    """What the environment needs from a data generator: core.data.burgers.Burgers, core.data.polynomial_pde.PolynomialPDE."""

    n_time: int  # snapshots per trajectory
    dt: float  # time between snapshots

    def ensemble(self, seed: int, n_batch: int) -> tuple[Field, Field]: ...


class Evaluation(Protocol):
    names: tuple[str, ...]
    score: float

    def describe(self) -> str: ...


E = TypeVar("E", bound=Evaluation)


def coarse(field: Field, level: int) -> Field:
    """The field filtered to |k| < pi / (2^level dx), on an odd grid with one point per kept mode."""
    return BlockCutoffFilter(block_size=2**level, odd_grid=True).transform(field.to_fourier()).to_real()


def read_only(a: np.ndarray) -> np.ndarray:
    """a, made read-only: the caches share their arrays with every caller."""
    a.flags.writeable = False
    return a


def extrapolate(values: np.ndarray, levels: np.ndarray, target: int) -> np.ndarray:
    """values (len(levels), ...) extrapolated linearly in the level to target: (...)."""
    slope, intercept = np.polyfit(levels, values.reshape(len(levels), -1), 1)
    return (slope * target + intercept).reshape(values.shape[1:])


class Environment(ABC, Generic[E]):
    # g is (batch, time, K, M) per basis and level, ~100 MB for M = 3 over the default levels: keep only the most recently used ones, which a fit, its noise and the candidate features share
    basis_modes_cache_size = 8
    # the noise is (batch, time - n_lags, K, M) per basis and level, ~75 MB per basis for M = 3 over the default fit levels: 32 entries ÷ 4 levels per basis = 8 bases.
    noise_cache_size = 32

    def __init__(
        self,
        dynamics: Dynamics | None = None,
        levels: Levels | None = None,
        n_fit: int = 400,
        n_score: int = 200,
        seed: int = 0,
        cost: float = 0.05,  # score cost per operator
        max_error: float = 2.0,  # rollout errors are capped here; a blow-up scores like an error of max_error
        n_lags: int = 4,  # memory of the Mori kernels: Omega^(0..n_lags-1), acting on the last n_lags states
    ):
        self.dynamics = Burgers() if dynamics is None else dynamics
        self.levels = Levels() if levels is None else levels
        self.cost, self.max_error = cost, max_error
        self.seed, self.n_fit, self.n_score = seed, n_fit, n_score  # fit trajectories from seed, scoring trajectories from seed + 1
        u, _ = self.dynamics.ensemble(seed, n_fit)
        u_score, _ = self.dynamics.ensemble(seed + 1, n_score)
        all_levels = self.levels.fit + (self.levels.predict,)
        self.u = {n: coarse(u, n) for n in all_levels}
        self.u_score = {self.levels.predict: coarse(u_score, self.levels.predict)}
        self.method = MoriKernels(homogeneous=True, n_lags=n_lags, origins="all")
        self._kernels: dict[tuple[frozenset, int], MemoryKernels] = {}
        self._operator_modes: dict[tuple[str, int], np.ndarray] = {}
        self._operator_power: dict[tuple[str, int], np.ndarray] = {}
        self._basis_modes: OrderedDict[tuple[tuple[str, ...], int], np.ndarray] = OrderedDict()
        self._gram_inverse: dict[tuple[tuple[str, ...], int], np.ndarray] = {}
        self._noise: OrderedDict[tuple[tuple[str, ...], int], np.ndarray] = OrderedDict()
        self.operators: dict[str, SymbolicOperator] = {}  # every operator seen, by canonical name
        self._evaluations: dict[frozenset, E] = {}

    # ---- the score ----

    def prepare(self, ops: list[SymbolicOperator]) -> list[SymbolicOperator]:
        """The basis as the environment uses it: without duplicates, in a canonical order."""
        unique = {op.name: op for op in ops}.values()
        return sorted(unique, key=lambda op: (op.degree, op.order, op.name))

    @abstractmethod
    def _evaluate(self, ops: list[SymbolicOperator]) -> E:
        """The evaluation of a prepared basis."""

    def evaluate(self, ops: list[SymbolicOperator]) -> E:
        ops = self.prepare(ops)
        key = frozenset(op.name for op in ops)
        if key not in self._evaluations:
            self._evaluations[key] = self._evaluate(ops)
        return self._evaluations[key]

    @property
    def history(self) -> list[E]:
        return list(self._evaluations.values())

    @property
    def n_evaluations(self) -> int:
        return len(self._evaluations)

    # ---- the Mori kernels of the observables g = (u, O_1(u), ...), the features of a basis ----

    def observable_ops(self, ops: list[SymbolicOperator]) -> list[SymbolicOperator]:
        """The operators besides u, which is always the first observable, in the canonical order."""
        return Environment.prepare(self, [op for op in ops if op.name != "u"])

    @staticmethod
    def basis(ops: list[SymbolicOperator]) -> OperatorBasis:
        return OperatorBasis([Identity(), *ops])

    def kernels(self, ops: list[SymbolicOperator], level: int) -> MemoryKernels:
        """Omega^(0..n_lags-1) at a level, fitted on the fit trajectories by MoriKernels, u and the operators; cached."""
        key = (frozenset(op.name for op in ops), level)
        if key not in self._kernels:
            g = self.basis_modes(ops, level)
            self._kernels[key] = self.method.fit_kernels(self.u[level], self.basis(ops), g=g)
        return self._kernels[key]

    def operator_modes(self, op: Operator, level: int) -> np.ndarray:
        """The modes of one operator O(u) on the fit trajectories at a level, (batch, time, K), in the order of the kernels; read-only, cached. op is the Identity, for u itself, or a SymbolicOperator."""
        key = (op.name, level)
        if isinstance(op, SymbolicOperator):
            self.operators.setdefault(op.name, op)
        if key not in self._operator_modes:
            self._operator_modes[key] = read_only(observables(op.evaluate(self.u[level]))[..., 0])
        return self._operator_modes[key]

    def operator_power(self, op: Operator, level: int) -> np.ndarray:
        """The power of the modes of one operator per time, sum over the batch of |O_k(t)|^2, on the fit trajectories at a level: (time, K). Read-only, cached."""
        key = (op.name, level)
        if key not in self._operator_power:
            self._operator_power[key] = read_only(np.sum(np.abs(self.operator_modes(op, level)) ** 2, axis=0))
        return self._operator_power[key]

    def basis_modes(self, ops: list[SymbolicOperator], level: int) -> np.ndarray:
        """The modes of every operator of the basis (Identity, *ops), the observables g = (u, O_1(u), ...), on the fit trajectories at a level: (batch, time, K, 1 + len(ops)), column i from operator_modes of operator i. ops as in kernels. Read-only, cached for the last basis_modes_cache_size bases and levels."""
        def stack() -> np.ndarray:
            return np.stack([self.operator_modes(op, level) for op in self.basis(ops).operators], axis=-1)

        return self._recent(self._basis_modes, (tuple(op.name for op in ops), level), self.basis_modes_cache_size, stack)

    def noise(self, ops: list[SymbolicOperator], level: int) -> np.ndarray:
        """W(t) = g(t) - sum_l Omega^(l) g(t-1-l) on the fit trajectories at a level, for every t after the first n_lags states: (batch, time - n_lags, K, M). ops as in kernels. Read-only, cached for the last noise_cache_size bases and levels."""
        def compute() -> np.ndarray:
            kernels, g = self.kernels(ops, level), self.basis_modes(ops, level)
            h = kernels.n_lags
            windows = sliding_window_view(g[:, :-1], h, axis=1)  # windows[b, s, k, j, w] = g_j(s + w) s = 0 … T-h-1,  w = 0 … h-1
            return g[:, h:] - np.einsum("lkij,bskjl->bski", kernels.kernels[::-1], windows, optimize=True)

        return self._recent(self._noise, (tuple(op.name for op in ops), level), self.noise_cache_size, compute)

    def gram_inverse(self, ops: list[SymbolicOperator], level: int) -> np.ndarray:
        """The pseudo-inverse of the Gram matrix <g_i, g_j> per wavevector of the observables g = (u, ops) on the fit trajectories at a level, over the batch and every time: (K, M, M). ops as in kernels. Read-only, cached."""
        key = (tuple(op.name for op in ops), level)
        if key not in self._gram_inverse:
            g = self.basis_modes(ops, level)
            self._gram_inverse[key] = read_only(covariance_inverse(np.einsum("btki,btkj->kij", g.conj(), g)))
        return self._gram_inverse[key]

    @staticmethod
    def _recent(cache: OrderedDict, key: Hashable, size: int, compute: Callable[[], np.ndarray]) -> np.ndarray:
        """cache[key], computed and made read-only on a miss; the cache keeps its `size` most recently used entries."""
        if key in cache:
            cache.move_to_end(key)
        else:
            cache[key] = read_only(compute())
            if len(cache) > size:
                cache.popitem(last=False)
        return cache[key]

    def one_step_residual(self, ops: list[SymbolicOperator], level: int) -> float:
        """|u_k(t+1) - sum_l Omega^(l)_u. g_k(t-l)| / |u_k(t+1) - u_k(t)| over the batch, steps and wavevectors."""
        return self._unexplained(self.noise(ops, level), self.basis_modes(ops, level))

    def _unexplained(self, W: np.ndarray, g: np.ndarray) -> float:
        h = g.shape[1] - W.shape[1]  # n_lags: W starts after the first n_lags states
        return float(np.linalg.norm(W[..., 0]) / np.linalg.norm(g[:, h:, :, 0] - g[:, h - 1:-1, :, 0]))

    def candidate_features(
        self, basis: list[SymbolicOperator], pool: list[SymbolicOperator], level: int
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        How much each candidate operator of a pool could add to a basis, from the Mori kernels of g = (u, basis), wavevector by wavevector:
        
        - relevance: the fraction of the noise W_u(t+1) of the u row that O(t) could explain as one more observable,
          sqrt(sum_k |<O_k, W_u,k>|^2 / |O_k|^2) / |W_u|;

        - spanned: the fraction of |O| in the span of g at equal times, the Mori projection with C(0);
        
        - unexplained: the one-step residual of u.
        """
        ops = self.observable_ops(basis)
        noise, g = self.noise(ops, level), self.basis_modes(ops, level)
        W_u, h = noise[..., 0], g.shape[1] - noise.shape[1]  # the u row of the noise, (batch, time - n_lags, K), and n_lags
        gram_inv = self.gram_inverse(ops, level)
        g_conj, W_u_conj, W_u_norm = g.conj(), W_u.conj(), np.linalg.norm(W_u)
        relevance, spanned = np.zeros(len(pool)), np.zeros(len(pool))
        for i, op in enumerate(pool):
            O, O_power = self.operator_modes(op, level), self.operator_power(op, level)  # (batch, time, K), (time, K)
            lagged_power = O_power[h - 1:-1].sum(axis=0)
            has_power = lagged_power > 1e-12 * max(lagged_power.max(), 1e-300)  # skips the wavevectors where O(t) is 0
            W_u_overlap = np.einsum("btk,btk->k", W_u_conj, O[:, h - 1:-1])  # <W_u(t+1), O(t)> per wavevector, (K,)
            relevance[i] = np.sqrt(np.sum(np.abs(W_u_overlap[has_power]) ** 2 / lagged_power[has_power])) / (W_u_norm + 1e-30)
            g_overlap = np.einsum("btki,btk->ki", g_conj, O)  # <g_i, O> per wavevector, (K, M)
            projected = np.real(np.einsum("ki,kij,kj->", g_overlap.conj(), gram_inv, g_overlap))
            spanned[i] = np.sqrt(max(projected, 0.0) / (O_power.sum() + 1e-30))
        return relevance, spanned, self._unexplained(noise, g)
