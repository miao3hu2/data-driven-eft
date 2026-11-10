import itertools

import numpy as np
import pytest

from core.data.base import Field
from core.data.burgers import Burgers
from core.evaluation.environment import Levels
from core.evaluation.kernels import KernelEnvironment
from core.inference.mori_kernels import observables
from core.operators.base import Identity
from core.operators.symbolic import SymbolicOperator, all_monomials

# a small ensemble: 64 points, levels 1 and 2 (31 and 15 points) to fit, level 3 (7 points) to predict
SMALL = Burgers(n=64, n_time=9)
LEVELS = Levels(fit=(1, 2), predict=3, n_extrapolation_levels=2)


def ops(*texts):
    return [SymbolicOperator(t) for t in texts]


@pytest.fixture(scope="module")
def kernel_env():
    return KernelEnvironment(SMALL, LEVELS, n_fit=60, n_score=20)


def test_burgers_derivatives_match_the_trajectories():
    # snapshots close enough for a central difference to resolve every mode (nu k^2 dt = 0.16 at the highest)
    burgers = Burgers(n=64, dt=0.002, n_time=5)
    u, dudt = burgers.ensemble(seed=0, n_batch=10)
    assert u.shape == dudt.shape == (10, 5, 64)
    assert u.time is not None
    np.testing.assert_allclose(u.time, np.arange(5) * 0.002)
    central = (u.values[:, 2:] - u.values[:, :-2]) / (2 * burgers.dt)
    error = np.linalg.norm(central - dudt.values[:, 1:-1]) / np.linalg.norm(dudt.values[:, 1:-1])
    assert error < 0.05


def test_burgers_ensemble_is_reproducible():
    a, _ = SMALL.ensemble(seed=3, n_batch=4)
    b, _ = SMALL.ensemble(seed=3, n_batch=4)
    np.testing.assert_array_equal(a.values, b.values)
    assert np.sqrt(np.mean(a.values[:, 0] ** 2)) == pytest.approx(SMALL.amplitude)


def test_symbolic_operators_are_defined_up_to_their_coupling():
    assert SymbolicOperator("D(u**2)").name == SymbolicOperator("2*u*D(u)").name


def test_kernel_env_linear_operators_only_cost(kernel_env):
    """u is always an observable and u_x is proportional to it mode by mode: same rollout, minus the cost."""
    alone, with_gradient = kernel_env.evaluate([]), kernel_env.evaluate(ops("D(u)"))
    assert with_gradient.error == pytest.approx(alone.error, rel=1e-8)
    assert with_gradient.score == pytest.approx(alone.score - kernel_env.cost, rel=1e-8)


def test_kernel_env_operators_are_defined_up_to_derivatives(kernel_env):
    """u u_x = (u^2)_x / 2 is proportional to u^2 at every wavevector."""
    square, advection = kernel_env.evaluate(ops("u**2")), kernel_env.evaluate(ops("u*D(u)"))
    assert advection.error == pytest.approx(square.error, rel=1e-8)


def test_kernel_env_u_alone_does_not_flow(kernel_env):
    """The modes of the prediction scale of u are the same at every level, and so are their kernels."""
    e = kernel_env.evaluate([])
    assert e.kernels.shape == (2, kernel_env.method.n_lags, 7, 1, 1)
    np.testing.assert_allclose(e.kernels[0], e.kernels[1], atol=1e-12)
    np.testing.assert_allclose(e.extrapolated, e.kernels[-1], atol=1e-12)


def test_kernel_env_ignores_u_and_caches(kernel_env):
    a = kernel_env.evaluate(ops("u**2"))
    assert kernel_env.evaluate(ops("u", "u**2")) is a
    assert a.names == ("u^2",)
    assert a in kernel_env.history
    assert "observables [u, u^2]" in a.describe()


def test_kernel_env_observables_come_from_the_cached_modes(kernel_env):
    """The observables assembled from the per-operator modes are those of the basis, and so are the kernels."""
    basis = kernel_env.observable_ops(ops("u**3", "u**2"))
    for level in LEVELS.fit:
        g = kernel_env.basis_modes(basis, level)
        np.testing.assert_allclose(g, observables(kernel_env.u[level], kernel_env.basis(basis)), atol=1e-10)
        direct = kernel_env.method.fit_kernels(kernel_env.u[level], kernel_env.basis(basis))
        np.testing.assert_allclose(kernel_env.kernels(basis, level).kernels, direct.kernels, atol=1e-8)


def test_kernel_env_caches_the_recent_basis_modes(kernel_env):
    """basis_modes returns the same array for a basis and level until enough other ones push it out."""
    basis = kernel_env.observable_ops(ops("u**2"))
    g = kernel_env.basis_modes(basis, 1)
    assert kernel_env.basis_modes(basis, 1) is g
    for cached in (g, kernel_env.operator_modes(basis[0], 1), kernel_env.operator_modes(Identity(), 1), kernel_env.operator_power(basis[0], 1)):
        with pytest.raises(ValueError, match="read-only"):
            cached[...] = 0
    texts = ["u**3", "D(u)**2", "u*D(D(u))", "D(u)**3"]
    others = [kernel_env.observable_ops(ops(*c)) for r in (1, 2) for c in itertools.combinations(texts, r)]
    assert len(others) >= kernel_env.basis_modes_cache_size
    for other in others:
        kernel_env.basis_modes(other, 1)
    assert len(kernel_env._basis_modes) == kernel_env.basis_modes_cache_size
    assert kernel_env.basis_modes(basis, 1) is not g


def test_kernel_env_noise_is_cached(kernel_env):
    """noise is g(t) minus the kernels on the last n_lags states, the same array until it is pushed out."""
    basis = kernel_env.observable_ops(ops("u**2", "u**3"))
    for level in LEVELS.fit:
        kernels, g = kernel_env.kernels(basis, level), kernel_env.basis_modes(basis, level)
        h = kernels.n_lags
        expected = np.stack([g[:, t] - kernels.predict(g[:, t - h : t]) for t in range(h, g.shape[1])], axis=1)
        W = kernel_env.noise(basis, level)
        np.testing.assert_allclose(W, expected, atol=1e-10)
        assert kernel_env.noise(basis, level) is W
        with pytest.raises(ValueError, match="read-only"):
            W[...] = 0


def test_kernel_env_operator_power_sums_the_modes_over_the_batch(kernel_env):
    op = ops("u**2")[0]
    power = kernel_env.operator_power(op, 1)
    np.testing.assert_allclose(power, np.sum(np.abs(kernel_env.operator_modes(op, 1)) ** 2, axis=0))
    assert kernel_env.operator_power(op, 1) is power


def test_kernel_env_gram_inverse_is_cached(kernel_env):
    """gram_inverse inverts <g_i, g_j> per wavevector, once per basis and level."""
    basis = kernel_env.observable_ops(ops("u**2"))
    g = kernel_env.basis_modes(basis, 1)
    inverse = kernel_env.gram_inverse(basis, 1)
    gram = np.einsum("btki,btkj->kij", g.conj(), g)
    np.testing.assert_allclose(gram @ inverse @ gram, gram, rtol=1e-8, atol=1e-8 * np.abs(gram).max())
    assert kernel_env.gram_inverse(basis, 1) is inverse
    with pytest.raises(ValueError, match="read-only"):
        inverse[...] = 0


def test_kernel_env_candidate_features_follow_the_observables(kernel_env):
    """Linear operators and the derivatives of an observable are spanned by the observables mode by mode; a missing
    nonlinearity is not, and is the most relevant to the u noise; adding it explains more of the u step."""
    pool = ops("D(u)", "D(u,2)", "u**2", "u*D(u)")
    relevance, spanned, unexplained = kernel_env.candidate_features([], pool, 1)
    np.testing.assert_allclose(spanned[:2], 1, rtol=1e-8)
    assert spanned[2] < 0.5 and relevance[2] > relevance[0]
    relevance, spanned, unexplained_with = kernel_env.candidate_features(ops("u**2"), pool, 1)
    np.testing.assert_allclose(spanned, 1, rtol=1e-8)
    assert unexplained_with < unexplained
    assert unexplained_with == pytest.approx(kernel_env.evaluate(ops("u**2")).residual[0])


def test_all_monomials_of_the_dsl_parse_and_are_distinct():
    texts = all_monomials()
    names = [SymbolicOperator(t).name for t in texts]
    assert len(set(names)) == len(names) and "u" in names
    assert len([t for t in texts if SymbolicOperator(t).degree > 1]) == 20


class Diffusion2D:
    """Exact anisotropic linear diffusion on a 2D periodic grid: u_k(t) = u_k(0) exp(-(a kx^2 + b ky^2) t)."""

    n, n_time, dt, a, b = 32, 8, 0.05, 0.3, 0.05

    def ensemble(self, seed, n_batch):
        dx = 2 * np.pi / self.n
        k = 2 * np.pi * np.fft.fftfreq(self.n, d=dx)
        kx, ky = np.meshgrid(k, k, indexing="ij")
        t = np.arange(self.n_time) * self.dt
        u0 = np.fft.fft2(np.random.default_rng(seed).standard_normal((n_batch, self.n, self.n)))
        u = np.fft.ifft2(u0[:, None] * np.exp(-(self.a * kx**2 + self.b * ky**2) * t[:, None, None])).real
        field = Field(values=u, scale=(dx, dx), batched=True, time=t)
        return field, field


def test_kernel_env_in_2d_matches_wavevectors_in_both_axes():
    # the decay differs along x and y, so matching the wavevectors of the prediction scale by one axis, or a wrong mode order in the
    # rollout, would put the wrong kernel on a mode
    dyn = Diffusion2D()
    env = KernelEnvironment(dyn, LEVELS, n_fit=20, n_score=10, cost=0.0)
    ev = env.evaluate([])
    k = env.k_predict
    assert k.shape == (9, 2)  # 32 points at level 3: 3 modes per axis
    exact = np.exp(-(dyn.a * k[:, 0] ** 2 + dyn.b * k[:, 1] ** 2) * dyn.dt)
    np.testing.assert_allclose(ev.direct[0, :, 0, 0], exact, atol=1e-10)
    np.testing.assert_allclose(ev.extrapolated[0, :, 0, 0], exact, atol=1e-10)
    assert ev.error < 1e-8
    assert "k=(1, 0)" in ev.describe()
