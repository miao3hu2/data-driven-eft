"""
A Burgers ensemble: random initial states evolved by du/dt = -(u^2/2)_x + nu u_xx on a periodic grid.
"""

from dataclasses import dataclass

import numpy as np

from core.data.base import Field


def random_initial_states(rng: np.random.Generator, n_batch: int, k: np.ndarray, k0: float, amplitude: float) -> np.ndarray:
    """Random phases on the spectrum k^2 / (1 + (k/k0)^2)^2 over the wavenumbers k of the grid, zero mean, rms
    amplitude: (n_batch, len(k))."""
    kk = np.abs(k)
    spectrum = (kk > 0) * kk**2 / (1 + (kk / k0) ** 2) ** 2
    amplitudes = rng.normal(size=(n_batch, len(k)))  # drawn before the phases: the order fixes the ensemble of a seed
    modes = np.sqrt(spectrum) * amplitudes * np.exp(2j * np.pi * rng.random((n_batch, len(k))))
    u = np.real(np.fft.ifft(modes, axis=-1))
    return amplitude * u / u.std()


@dataclass(frozen=True)
class Burgers:
    n: int = 256  # grid points
    length: float = 2 * np.pi
    nu: float = 0.08
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

    def random_initial_states(self, rng: np.random.Generator, n_batch: int) -> np.ndarray:
        """Random phases on the spectrum k^2 / (1 + (k/k0)^2)^2, zero mean, rms amplitude: (n_batch, n)."""
        return random_initial_states(rng, n_batch, self.k, self.k0, self.amplitude)

    def rhs(self, uk: np.ndarray) -> np.ndarray:
        """d/dt u_k, dealiased by the 2/3 rule."""
        k = self.k
        dealias = np.abs(k) < (2 / 3) * np.abs(k).max()
        u = np.real(np.fft.ifft(uk, axis=-1))
        return -0.5j * k * np.fft.fft(u * u, axis=-1) * dealias - self.nu * k**2 * uk

    def solve(self, u0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """States and exact time derivatives at the snapshots: (batch, n_time, n) each."""
        uk = np.fft.fft(u0, axis=-1)
        u, dudt = [], []
        for step in range(self.n_time):
            u.append(np.real(np.fft.ifft(uk, axis=-1)))
            dudt.append(np.real(np.fft.ifft(self.rhs(uk), axis=-1)))
            if step == self.n_time - 1:
                break
            h = self.h
            for _ in range(round(self.dt / h)):
                k1 = self.rhs(uk)
                k2 = self.rhs(uk + h / 2 * k1)
                k3 = self.rhs(uk + h / 2 * k2)
                k4 = self.rhs(uk + h * k3)
                uk = uk + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return np.stack(u, axis=1), np.stack(dudt, axis=1)

    def ensemble(self, seed: int, n_batch: int) -> tuple[Field, Field]:
        """n_batch trajectories from random initial states: the states and their time derivatives, batched Fields."""
        u, dudt = self.solve(self.random_initial_states(np.random.default_rng(seed), n_batch))
        as_field = lambda v: Field(values=v, scale=self.dx, batched=True, time=self.times)
        return as_field(u), as_field(dudt)
