"""
Ensembles of a 1D polynomial PDE du/dt = F(u) on a periodic grid, F written in the operator language of
core.operators.symbolic with its coefficients, e.g.

    PolynomialPDE("-0.5*D(u**2) + 0.08*D(u,2)")      Burgers
    PolynomialPDE("-D(u**3)/3 + 0.08*D(u,2)")        modified Burgers
    PolynomialPDE("-0.5*D(u)**2 + 0.08*D(u,2)")      KPZ-like growth

The terms linear in u (u, u_x, ..., u_xxxx with their coefficients) form a diagonal operator L(k) in Fourier space and
are integrated exactly by an integrating factor; the nonlinear ones are evaluated like a SymbolicOperator, derivatives
spectrally and products dealiased on a grid twice as fine, and stepped with RK4. Initial states and the interface
(ensemble, n_time, dt) are those of core.data.burgers.Burgers, so a PolynomialPDE can replace it in core.evaluation
environments.
"""

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from core.data.base import Field
from core.data.burgers import random_initial_states
from core.operators.symbolic import Monomial, evaluate_monomials, parse


@dataclass(frozen=True)
class PolynomialPDE:
    rhs: str  # F(u) in the DSL, with its coefficients
    n: int = 256  # grid points
    length: float = 2 * np.pi
    dt: float = 0.05  # time between snapshots
    n_time: int = 21  # snapshots, the first one the initial state
    h: float = 1e-3  # RK4 step
    k0: float = 3.0  # peak wavenumber of the initial energy spectrum k^2 / (1 + (k/k0)^2)^2
    amplitude: float = 1.2  # rms of the initial states over the ensemble

    @property
    def dx(self) -> float:
        return self.length / self.n

    @property
    def k(self) -> np.ndarray:
        return 2 * np.pi * np.fft.fftfreq(self.n, d=self.dx)

    @property
    def times(self) -> np.ndarray:
        return np.arange(self.n_time) * self.dt

    @cached_property
    def monomials(self) -> list[Monomial]:
        monomials = parse(self.rhs, normalize=False)
        if any(m.n_space > 1 for m in monomials):
            raise ValueError(f"PolynomialPDE is 1D: {self.rhs!r} differentiates along y or z.")
        return monomials

    @cached_property
    def nonlinear(self) -> list[Monomial]:
        return [m for m in self.monomials if m.degree > 1]

    @cached_property
    def linear(self) -> np.ndarray:
        """L(k), the Fourier symbol of the linear terms, 0 at the Nyquist mode of an even grid (which has no sign)."""
        symbol = sum((m.coefficient * (1j * self.k) ** m.order for m in self.monomials if m.degree == 1), np.zeros(self.n, dtype=complex))
        return symbol * self._kept

    @cached_property
    def _kept(self) -> np.ndarray:
        kept = np.ones(self.n)
        if self.n % 2 == 0:
            kept[self.n // 2] = 0
        return kept

    def random_initial_states(self, rng: np.random.Generator, n_batch: int) -> np.ndarray:
        return random_initial_states(rng, n_batch, self.k, self.k0, self.amplitude)

    def nonlinear_rhs(self, uk: np.ndarray) -> np.ndarray:
        """The nonlinear terms of F in Fourier space."""
        if not self.nonlinear:
            return np.zeros_like(uk)
        u = np.real(np.fft.ifft(uk, axis=-1))
        return np.fft.fft(evaluate_monomials(self.nonlinear, u, self.dx), axis=-1) * self._kept

    def solve(self, u0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """States and exact time derivatives at the snapshots: (batch, n_time, n) each. Integrating-factor RK4."""
        h, L = self.h, self.linear
        E, E2 = np.exp(L * h), np.exp(L * h / 2)
        uk = np.fft.fft(u0, axis=-1) * self._kept
        u, dudt = [], []
        for step in range(self.n_time):
            u.append(np.real(np.fft.ifft(uk, axis=-1)))
            dudt.append(np.real(np.fft.ifft(L * uk + self.nonlinear_rhs(uk), axis=-1)))
            if step == self.n_time - 1:
                break
            for _ in range(round(self.dt / h)):
                k1 = self.nonlinear_rhs(uk)
                k2 = self.nonlinear_rhs(E2 * (uk + h / 2 * k1))
                k3 = self.nonlinear_rhs(E2 * uk + h / 2 * k2)
                k4 = self.nonlinear_rhs(E * uk + h * E2 * k3)
                uk = E * uk + h / 6 * (E * k1 + 2 * E2 * (k2 + k3) + k4)
        return np.stack(u, axis=1), np.stack(dudt, axis=1)

    def ensemble(self, seed: int, n_batch: int) -> tuple[Field, Field]:
        """n_batch trajectories from random initial states: the states and their time derivatives, batched Fields."""
        u, dudt = self.solve(self.random_initial_states(np.random.default_rng(seed), n_batch))
        as_field = lambda v: Field(values=v, scale=self.dx, batched=True, time=self.times)
        return as_field(u), as_field(dudt)
