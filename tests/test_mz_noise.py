import numpy as np
import pytest
import torch

from core.operators.symbolic import SymbolicOperator
from models.mz_noise import evaluate, fit_mean, fit_rnn_noise, orthogonal_noise, simulate
from models.orthogonal_dynamics import MeanNetwork
from models.stochastic_rnn import StochasticRNN
from posttrain.tasks import EnvSize, environment, task

TINY = EnvSize(n=64, n_time=9, fit=(1, 2), heldout=3, n_train=20, n_test=10)


def _net():
    x = torch.randn(3, 8, 7, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    return MeanNetwork(8), x


def test_the_mean_network_draws_nothing():
    net, x = _net()
    state = torch.get_rng_state()
    first = net(x, 6)
    assert torch.equal(torch.get_rng_state(), state)
    torch.manual_seed(123)
    assert torch.equal(net(x, 6), first)


def test_the_hidden_state_stays_bounded_whatever_the_weights():
    net, x = _net()
    with torch.no_grad():
        for p in net.parameters():
            p.mul_(50)
        h = net.initial_state(x)
        for _ in range(200):
            h = net.evolve(h)
            assert h.abs().max() <= 1


def test_the_mean_is_translation_equivariant():
    net, x = _net()
    assert torch.allclose(net(x.roll(2, dims=-1), 5), net(x, 5).roll(2, dims=-1), atol=1e-12)


@pytest.fixture(scope="module")
def closure():
    env = environment(task("burgers"), TINY)
    return env, env.extrapolated_kernels([SymbolicOperator("u**2")])


def test_the_deterministic_simulation_is_the_closure_the_loop_scores(closure):
    env, kernels = closure
    level, h = env.levels.heldout, kernels.n_lags
    stats = evaluate(env.u_test[level], kernels, {}, n_samples=1)["no_noise"]
    assert stats["error"] == pytest.approx(env.rollout_error(kernels, level), rel=1e-10)
    assert stats["spread_skill"] == 0 and stats["blowups"] == 0
    path = simulate(env.u_test[level], kernels)[0]
    np.testing.assert_allclose(path[:, :h], env.u_test[level].values[:, :h])


def test_the_noise_is_what_the_drift_leaves(closure):
    env, kernels = closure
    u = env.u[env.levels.heldout]
    g, W = orthogonal_noise(u, kernels)
    h = kernels.n_lags
    assert W.shape == (u.values.shape[0], u.values.shape[1] - h, g.shape[2])
    predicted = kernels.predict(g[:, :h])[..., 0]  # the step to time h from the first h states
    np.testing.assert_allclose(W[:, 0], g[:, h, :, 0] - predicted, atol=1e-12)


def _time_dependent_rnn(real: bool):
    """One mode, two hidden units, with A and sigma changing in time; its parameters and the covariance of W in closed form."""
    rnn = StochasticRNN(1, 5, torch.tensor([real]), d_h=2, degree_sigma=2, degree_A=1, seed=0)
    dtype = np.float64 if real else np.complex128
    values = {
        "A0": np.array([[0.5, 0.1], [0.0, 0.4]], dtype=dtype), "A1": np.array([[0.1, 0.0], [0.05, -0.1]], dtype=dtype),
        "S": np.array([[0.4, 0.0], [0.1, 0.3]], dtype=dtype), "S0": np.array([[1.1, 0.2], [0.0, 0.7]], dtype=dtype),
        "c": np.array([0.6, 0.3], dtype=dtype), "s": np.array([-0.3, 0.5, -1.0]), "r": 0.15,
    }
    if not real:
        values["A0"] = values["A0"] * np.exp(0.3j)
        values["S"] = values["S"] * np.exp(-0.2j)
        values["c"] = values["c"] * np.array([np.exp(0.4j), 1])
    with torch.no_grad():
        rnn.A[0, 0] = torch.as_tensor(values["A0"], dtype=torch.complex128)
        rnn.A[1, 0] = torch.as_tensor(values["A1"], dtype=torch.complex128)
        rnn.S[0] = torch.as_tensor(values["S"], dtype=torch.complex128)
        rnn.S0[0] = torch.as_tensor(values["S0"], dtype=torch.complex128)
        rnn.c[0] = torch.as_tensor(values["c"], dtype=torch.complex128)
        rnn.log_sigma[:, 0] = torch.as_tensor(values["s"])
        rnn.log_r.fill_(np.log(values["r"]))
    T, c = 5, values["c"]
    As = [rnn.A_at(n)[0].detach().numpy() for n in range(T)]
    sigmas = [rnn.sigma_at(n)[0].detach().item() for n in range(T)]
    # W(n) = sum over the sources of their gains: the gains of xi0, xi(0..T-1) and eps(0..T-1) in W(0..T-1)
    gains = np.zeros((T, 2 + 2 * T + T), dtype=complex)
    state = np.zeros((2, gains.shape[1]), dtype=complex)
    state[:, :2] = values["S0"]
    for n in range(T):
        state = As[n] @ state
        state[:, 2 + 2 * n:4 + 2 * n] += sigmas[n] * values["S"]
        gains[n] = c.conj() @ state
        gains[n, 2 + 2 * T + n] += sigmas[n] * np.sqrt(values["r"])
    return rnn, gains @ gains.conj().T


def test_kalman_likelihood_with_time_dependent_coefficients_is_exact_on_a_real_mode():
    rnn, cov = _time_dependent_rnn(real=True)
    cov = cov.real
    W = np.random.default_rng(0).multivariate_normal(np.zeros(5), cov, size=3)
    exact = np.mean([0.5 * (w @ np.linalg.solve(cov, w) + np.linalg.slogdet(2 * np.pi * cov)[1]) for w in W]) / 5
    assert rnn.nll(torch.as_tensor(W[..., None], dtype=torch.complex128)).item() == pytest.approx(exact, rel=1e-10)


def test_kalman_likelihood_with_time_dependent_coefficients_is_exact_on_a_complex_mode():
    rnn, cov = _time_dependent_rnn(real=False)
    rng = np.random.default_rng(0)
    L = np.linalg.cholesky(cov)
    W = (L @ ((rng.normal(size=(3, 5)) + 1j * rng.normal(size=(3, 5))) / np.sqrt(2))[..., None])[..., 0]
    exact = np.mean([(w.conj() @ np.linalg.solve(cov, w)).real + np.linalg.slogdet(np.pi * cov)[1] for w in W]) / 5
    assert rnn.nll(torch.as_tensor(W[..., None], dtype=torch.complex128)).item() == pytest.approx(exact, rel=1e-10)


def test_the_generated_covariance_is_the_one_of_the_likelihood():
    rnn, cov = _time_dependent_rnn(real=False)
    W = rnn.path(rnn.latent(40000, 5, torch.Generator().manual_seed(0)))[..., 0].numpy()
    np.testing.assert_allclose(W.T @ W.conj() / len(W), cov, atol=0.05)


def test_given_the_latent_the_path_of_the_rnn_is_deterministic():
    rnn, _ = _time_dependent_rnn(real=False)
    z = rnn.latent(2, 5, torch.Generator().manual_seed(1))
    state = torch.get_rng_state()
    first = rnn.path(z)
    assert torch.equal(torch.get_rng_state(), state)  # nothing is drawn along the path
    torch.manual_seed(123)
    assert torch.equal(rnn.path(z), first)
    assert (rnn.path(rnn.latent(2, 5, torch.Generator().manual_seed(2))) - first).abs().max() > 0


def test_the_rnn_closure_is_fixed_by_the_initial_states_and_the_latent(closure):
    env, kernels = closure
    u = env.u_test[env.levels.heldout]
    h = kernels.n_lags
    fitted = fit_rnn_noise(env.u[env.levels.heldout], kernels, degree_A=1, iters=30, mean_options=dict(folds=2, iters=5))
    assert fitted.losses[-1] < fitted.losses[0]
    W = fitted.sample(np.zeros((2, h, u.values.shape[-1], 2)), 4, torch.Generator().manual_seed(0))
    n = W.shape[-1]
    np.testing.assert_allclose(W[..., -np.arange(n) % n], W.conj(), atol=1e-12)
    later = u.values.copy()
    later[:, h:] += 1.0  # the same initial states, a different future
    a = simulate(u, kernels, fitted, n_samples=2, seed=3)
    np.testing.assert_array_equal(a, simulate(u.replace(values=later), kernels, fitted, n_samples=2, seed=3))
    assert np.abs(a[0] - a[1])[:, h:].max() > 0
    assert evaluate(u, kernels, {"rnn": fitted}, n_samples=2)["rnn"]["blowups"] == 0


def test_the_mean_of_the_noise_is_cross_fitted_and_fixed_by_the_initial_states(closure):
    env, kernels = closure
    train, u = env.u[env.levels.heldout], env.u_test[env.levels.heldout]
    h = kernels.n_lags
    fitted = fit_rnn_noise(train, kernels, iters=10, mean_options=dict(folds=2, iters=5))
    mean = fitted.mean
    assert len(mean.nets) == 2 and np.all((mean.shrink >= 0) & (mean.shrink <= 1))
    g, W = orthogonal_noise(train, kernels)
    _, residual = fit_mean(g, W, kernels, folds=2, iters=5)
    in_sample = W - mean.predict(g[:, :h], W.shape[1])
    assert residual.shape == W.shape and np.abs(residual - in_sample).max() > 0  # not the residual of the final model
    first = mean.predict(g[:, :h], 4)
    n = first.shape[-1]
    np.testing.assert_allclose(first[..., -np.arange(n) % n], first.conj(), atol=1e-12)
    torch.manual_seed(5)
    np.testing.assert_array_equal(mean.predict(g[:, :h], 4), first)  # deterministic
    later = u.values.copy()
    later[:, h:] += 1.0
    np.testing.assert_array_equal(simulate(u, kernels, fitted, n_samples=2, seed=3),
                                  simulate(u.replace(values=later), kernels, fitted, n_samples=2, seed=3))
