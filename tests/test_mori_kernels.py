import numpy as np
import pytest
import scipy.linalg

from core.coarse_graining.filter import BlockCutoffFilter
from core.data.base import Field
from core.inference.mori_kernels import (
    MoriKernels,
    memory_kernels,
    mori_kernels_level,
    observables,
    two_time_correlations,
)
from core.operators.base import Identity, OperatorBasis
from core.rgflow.memory import mori_kernel_data

# du/dt = NU u_xx + A cos(x) u_x on an odd grid, so that every mode is kept (no Nyquist mode)
N = 15
DX = 2 * np.pi / N
X = np.arange(N) * DX
K = 2 * np.pi * np.fft.fftfreq(N, d=DX)
NU, A, DT = 0.1, 1.0, 0.2
COARSE = BlockCutoffFilter(block_size=2, odd_grid=True)  # keeps |k| <= 3: 7 of the 15 modes
COARSE_MODES = np.array([0, 1, 2, 3, N - 3, N - 2, N - 1])  # their indices in the length-N spectrum, in kept order


def _step(a=A):
    """The exact propagator over DT in real space."""
    F = np.fft.fft(np.eye(N), axis=0)
    d1 = np.real(np.linalg.solve(F, np.diag(1j * K) @ F))
    d2 = np.real(np.linalg.solve(F, np.diag(-(K**2)) @ F))
    return scipy.linalg.expm(DT * (NU * d2 + a * np.diag(np.cos(X)) @ d1))


def _fourier_step(a=A):
    """The same propagator acting on the Fourier modes: M = F step F^-1."""
    F = np.fft.fft(np.eye(N), axis=0)
    return F @ _step(a) @ np.linalg.inv(F)


def _trajectories(rng, n_batch, n_time, a=A):
    """White-noise initial states, evolved exactly: (batch, time, N)."""
    step = _step(a)
    u = [rng.normal(size=(n_batch, N))]
    for _ in range(n_time - 1):
        u.append(u[-1] @ step.T)
    return Field(values=np.stack(u, axis=1), scale=DX, batched=True, time=np.arange(n_time) * DT)


def _exact_kernels(n_lags):
    """Population kernels for the coarse modes with white-noise initial states (C(0) = I / N, so the Mori projection
    is the orthogonal projection P = R^H R onto the kept modes): Omega^(l) = R (M Q)^l M R^H with Q = 1 - P."""
    M = _fourier_step()
    R = np.eye(N)[COARSE_MODES]
    MQ = M @ (np.eye(N) - R.T @ R)
    return np.stack([R @ np.linalg.matrix_power(MQ, l) @ M @ R.T for l in range(n_lags)])


def test_resolved_advection_is_markovian():
    """With every mode observed, Omega^(0) is the exact propagator and there is no memory."""
    rng = np.random.default_rng(0)
    kernels = mori_kernels_level(_trajectories(rng, n_batch=200, n_time=6), homogeneous=False)

    assert kernels.kernels.shape == (5, 1, N, N)
    np.testing.assert_allclose(kernels.kernels[0, 0], _fourier_step(), atol=1e-10)
    np.testing.assert_allclose(kernels.kernels[1:], 0, atol=1e-10)


def test_recursion_on_exact_correlations():
    """Fed the population correlations C(n) = R M^n R^H / N of the coarse modes, the recursion gives the exact
    Mori-Zwanzig kernels, with memory, since cos(x) couples the kept modes to the dropped ones."""
    n_lags = 6
    M = _fourier_step()
    R = np.eye(N)[COARSE_MODES]
    C = np.stack([R @ np.linalg.matrix_power(M, n) @ R.T / N for n in range(n_lags + 1)])[:, None]

    omega = memory_kernels(C)[:, 0]
    exact = _exact_kernels(n_lags)
    np.testing.assert_allclose(omega, exact, atol=1e-12)
    assert np.linalg.norm(exact[1]) > 0.01 * np.linalg.norm(exact[0])


def test_coarse_advection_matches_exact_kernels():
    """On sampled coarse trajectories, every kernel approaches the population one, relative to its own size: the
    memory kernels are a few percent of the Markov term, so a fit without memory would be off by 100% there."""
    rng = np.random.default_rng(1)
    coarse = COARSE.transform(_trajectories(rng, n_batch=40000, n_time=5)).to_real()
    kernels = mori_kernels_level(coarse, homogeneous=False)

    exact = _exact_kernels(4)
    for l in range(4):
        error = np.linalg.norm(kernels.kernels[l, 0] - exact[l]) / np.linalg.norm(exact[l])
        assert error < 0.1, f"Omega^({l}): relative error {error:.3f}"


def test_noise_is_orthogonal_to_initial_observables():
    rng = np.random.default_rng(2)
    coarse = COARSE.transform(_trajectories(rng, n_batch=300, n_time=7)).to_real()
    kernels = mori_kernels_level(coarse, homogeneous=False)
    g = observables(coarse, homogeneous=False)

    W = kernels.noise(g)
    assert W.shape == (300, 6, 1, 7)
    overlap = np.einsum("bnki,bkj->nkij", W, g[:, 0].conj()) / len(g)
    np.testing.assert_allclose(overlap, 0, atol=1e-10 * np.abs(kernels.correlations[0]).max())

    # without the noise, the equation regresses g(n) on g(0)
    C = kernels.correlations
    regression = np.einsum("nkij,kjl,bkl->bnki", C, np.linalg.inv(C[0]), g[:, 0])
    np.testing.assert_allclose(kernels.rollout(g[:, 0], 6), regression, atol=1e-10)


def test_uniform_advection_is_markovian_mode_by_mode():
    """du/dt = NU u_xx - A u_x is translation invariant: the modes decouple, so even the coarse modes have no memory
    and Omega^(0)_k = exp(DT (-NU k^2 - i A k))."""
    rng = np.random.default_rng(3)
    F = np.fft.fft(np.eye(N), axis=0)
    step = np.real(np.linalg.solve(F, np.diag(np.exp(DT * (-NU * K**2 - 1j * A * K))) @ F))
    u = [rng.normal(size=(200, N))]
    for _ in range(4):
        u.append(u[-1] @ step.T)
    field = Field(values=np.stack(u, axis=1), scale=DX, batched=True, time=np.arange(5) * DT)
    kernels = mori_kernels_level(COARSE.transform(field).to_real(), homogeneous=True)

    assert kernels.kernels.shape == (4, 7, 1, 1)
    np.testing.assert_allclose(kernels.wavevectors[:, 0], K[COARSE_MODES])
    np.testing.assert_allclose(kernels.kernels[0, :, 0, 0], np.exp(DT * (-NU * K**2 - 1j * A * K))[COARSE_MODES], atol=1e-10)
    np.testing.assert_allclose(kernels.kernels[1:], 0, atol=1e-10)


def test_correlations_over_all_origins():
    rng = np.random.default_rng(4)
    g = rng.normal(size=(10, 6, 3, 2)) + 1j * rng.normal(size=(10, 6, 3, 2))
    C = two_time_correlations(g, n_lags=2, origins="all")

    direct = np.zeros((3, 3, 2, 2), dtype=complex)
    for s in range(4):
        for n in range(3):
            direct[n] += np.einsum("bki,bkj->kij", g[:, s + n], g[:, s].conj()) / 40
    np.testing.assert_allclose(C, direct)
    with pytest.raises(ValueError, match="n_lags"):
        two_time_correlations(g, n_lags=6)


def test_inference_method_and_levels():
    rng = np.random.default_rng(5)
    field = _trajectories(rng, n_batch=100, n_time=4)
    basis = OperatorBasis([Identity()])

    markov = MoriKernels(homogeneous=True).fit(field, basis)
    assert markov.names == ["I"] and markov["I"].shape == (N, 1, 1)
    markov = MoriKernels(homogeneous=False).fit(field, basis)
    assert markov["I"].shape == (N, N, 1)
    np.testing.assert_allclose(markov["I"][:, :, 0], _fourier_step(), atol=1e-10)

    data = mori_kernel_data(field, n_levels=2, homogeneous=False)
    assert [d.level for d in data] == [1, 2]
    assert [d.kernels.shape[-1] for d in data] == [7, 3]
    assert data[0].dt == pytest.approx(DT)


def test_rejects_unequal_time_steps():
    rng = np.random.default_rng(6)
    field = _trajectories(rng, n_batch=10, n_time=3)
    with pytest.raises(ValueError, match="equally spaced"):
        mori_kernels_level(field.replace(time=np.array([0.0, 1.0, 3.0])))
