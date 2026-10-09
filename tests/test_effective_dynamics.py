"""Inference of coarse dynamics: MoriZwanzig and MemoryRatio on trajectories with known dynamics."""

import numpy as np
import pytest

from core.data.base import Field
from core.effective_theory.base import EffectiveDynamics
from core.inference.base import infer_couplings
from core.inference.mori_zwanzig import MemoryRatio, MoriZwanzig, mori_zwanzig_level
from core.operators.base import Identity, Operator, OperatorBasis
from core.rgflow.memory import mori_zwanzig_data


class _Square(Operator):
    name = "u2"

    def evaluate(self, field):
        return field.replace(values=field.values ** 2)


def _decaying(rng, n_batch=200, n_time=5, n=8):
    """u(t+1) = exp(-|k|^2 / 4) u(t) mode by mode."""
    k = 2 * np.pi * np.fft.fftfreq(n)
    decay = np.exp(-(k[:, None] ** 2 + k[None, :] ** 2) / 4)
    u = [rng.normal(size=(n_batch, n, n))]
    for _ in range(n_time - 1):
        u.append(np.real(np.fft.ifft2(decay * np.fft.fft2(u[-1]))))
    return Field(values=np.stack(u, axis=1), scale=1.0, batched=True, time=np.arange(n_time))


def _pointwise(rng, a=-0.3, b=0.2, n_batch=300, n_time=5, n=7):
    """u(t+1) = u(t) + a u(t) + b u(t)^2 at every grid point."""
    u = [0.3 * rng.normal(size=(n_batch, n, n))]
    for _ in range(n_time - 1):
        u.append(u[-1] + a * u[-1] + b * u[-1] ** 2)
    return Field(values=np.stack(u, axis=1), scale=1.0, batched=True, time=np.arange(n_time))


def test_markov_couplings_are_the_exact_linear_dynamics_per_wavevector():
    d = mori_zwanzig_data(_decaying(np.random.default_rng(0)), n_levels=1)[0]
    k2 = np.sum(d.wavevectors ** 2, axis=1)
    np.testing.assert_allclose(d.markov["I"][:, 0], np.exp(-k2 / 4) - 1, atol=1e-10)
    np.testing.assert_allclose(d.memory_couplings["I"], 0, atol=1e-10)


def test_nonlinear_dynamics_are_recovered_with_a_nonlinear_basis():
    field = _pointwise(np.random.default_rng(1))
    basis = OperatorBasis([Identity(), _Square()])
    dynamics = MoriZwanzig().fit_dynamics(field, basis)

    assert isinstance(dynamics, EffectiveDynamics)
    np.testing.assert_allclose(dynamics.markov["I"], -0.3, atol=1e-8)
    np.testing.assert_allclose(dynamics.markov["u2"], 0.2, atol=1e-8)
    np.testing.assert_allclose(dynamics.drift(1)["u2"], 0.2 + dynamics.ratio["gamma"][1] * dynamics.memory["u2"])
    assert dynamics.describe().startswith("du = (")


def test_a_linear_basis_leaves_the_nonlinearity_to_the_residual():
    field = _pointwise(np.random.default_rng(2))
    linear = mori_zwanzig_level(field)
    nonlinear = mori_zwanzig_level(field, OperatorBasis([Identity(), _Square()]))
    # u^2 is not a linear function of the modes of u[0], so only the larger basis explains the step at t = 0
    assert np.abs(linear.residual.values[:, 0]).max() > 1e-3
    np.testing.assert_allclose(nonlinear.residual.values[:, 0], 0, atol=1e-8)


def test_inference_methods_share_the_couplings_interface():
    field = _pointwise(np.random.default_rng(3))
    basis = OperatorBasis([Identity()])
    level = mori_zwanzig_level(field, basis)

    markov = infer_couplings(field, basis, MoriZwanzig())
    np.testing.assert_allclose(markov["I"], level.markov["I"])
    ratio = infer_couplings(field, basis, MemoryRatio())
    gamma, stderr = level.ratio()
    np.testing.assert_allclose(ratio["gamma"], gamma)
    assert ratio.stderr is not None
    np.testing.assert_allclose(ratio.stderr["gamma"], stderr)


@pytest.mark.parametrize("homogeneous", [True, False])
def test_dynamics_collects_the_level(homogeneous):
    rng = np.random.default_rng(4)
    field = Field(values=rng.normal(size=(200, 5, 8, 8)), scale=1.0, batched=True, time=np.arange(5))
    d = mori_zwanzig_data(field, n_levels=1, homogeneous=homogeneous)[0]
    dynamics = d.dynamics()

    assert dynamics.basis.names == ["I"] and dynamics.homogeneous == homogeneous
    np.testing.assert_array_equal(dynamics.time, [0, 1, 2])
    np.testing.assert_array_equal(dynamics.wavevectors, d.wavevectors)
    assert dynamics.wavevectors.shape == (9, 2)
    np.testing.assert_allclose(dynamics.ratio["gamma"], d.ratio()[0])
    assert dynamics.noise["noise_var"].shape == (3, 3, 3)
    np.testing.assert_allclose(dynamics.noise["noise_var"], d.noise_var())
    expected = (9, 1) if homogeneous else (9, 1, 3, 3)
    assert dynamics.markov["I"].shape == dynamics.memory["I"].shape == expected


@pytest.mark.parametrize("homogeneous", [True, False])
def test_evaluate_gives_the_drift_of_the_fitted_terms(homogeneous):
    rng = np.random.default_rng(5)
    field = Field(values=rng.normal(size=(300, 5, 5, 5)), scale=1.0, batched=True, time=np.arange(5))
    d = mori_zwanzig_level(field, homogeneous=homogeneous)
    dynamics = d.dynamics()
    states = field.replace(values=field.values[:, :3], time=np.arange(3))
    increment = field.values[:, 1:4] - field.values[:, :3]

    # the Markov drift P L u = L u - Q L u, at every state
    np.testing.assert_allclose(dynamics.evaluate(states).values, increment - d.residual.values, atol=1e-10)
    # with the memory: P L u + gamma(t) P L Q L u at time step t
    for t in range(3):
        at_t = field.replace(values=field.values[:, t], time=None)
        expected = increment[:, t] - d.residual.values[:, t] + d.ratio()[0][t] * d.memory.values[:, t]
        np.testing.assert_allclose(dynamics.evaluate(at_t, t=t).values, expected, atol=1e-10)
    # the same with gamma taken at the time steps of the field, also on time steps that do not start at the first
    expected = increment - d.residual.values + d.ratio()[0][None, :, None, None] * d.memory.values
    np.testing.assert_allclose(dynamics.evaluate(states, memory=True).values, expected, atol=1e-10)
    later = field.replace(values=field.values[:, 1:3], time=np.arange(1, 3))
    np.testing.assert_allclose(dynamics.evaluate(later, memory=True).values, expected[:, 1:3], atol=1e-10)


def test_evaluate_takes_gamma_at_the_time_steps_of_the_field():
    rng = np.random.default_rng(5)
    field = Field(values=rng.normal(size=(300, 5, 5, 5)), scale=1.0, batched=True, time=np.arange(5))
    dynamics = mori_zwanzig_level(field).dynamics()
    states = field.replace(values=field.values[:, :3], time=np.arange(3))
    with pytest.raises(ValueError, match="memory=True"):
        dynamics.evaluate(states, t=0)
    with pytest.raises(ValueError, match="time steps"):
        dynamics.evaluate(field.replace(time=np.arange(5) + 0.5), memory=True)
    with pytest.raises(ValueError, match="index t"):
        dynamics.evaluate(field.replace(values=field.values[:, 0], time=None), memory=True)


def test_evaluate_reproduces_recovered_nonlinear_dynamics():
    field = _pointwise(np.random.default_rng(6))
    dynamics = MoriZwanzig().fit_dynamics(field, OperatorBasis([Identity(), _Square()]))
    u0 = field.replace(values=field.values[:, 0], time=None)
    np.testing.assert_allclose(dynamics.evaluate(u0).values, field.values[:, 1] - field.values[:, 0], atol=1e-8)


def test_evaluate_needs_the_grid_of_the_dynamics():
    field = _pointwise(np.random.default_rng(7))
    dynamics = MoriZwanzig().fit_dynamics(field, OperatorBasis([Identity()]))
    with pytest.raises(ValueError, match="grid"):
        dynamics.evaluate(Field(values=np.zeros((2, 9, 9)), scale=1.0, batched=True))
