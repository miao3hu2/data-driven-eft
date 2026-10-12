import dataclasses
import warnings

import numpy as np
import pytest

from core.coarse_graining.filter import BlockCutoffFilter
from core.data.base import Field
from core.inference.mori_zwanzig import MoriZwanzigData
from core.operators.mori_zwanzig import Projection
from core.rgflow.memory import NoiseData, RatioData, mori_zwanzig_data, noise_data, ratio_data


def _random_trajectories(rng, n_batch=200, n_time=5, n=8):
    return Field(values=rng.normal(size=(n_batch, n_time, n, n)), scale=1.0, batched=True, time=np.arange(n_time))


def _linear_trajectories(rng, n_batch=200, n_time=5, n=8):
    """u(t+1) = exp(-|k|^2 / 4) u(t) mode by mode: Markovian dynamics that every filter level resolves exactly."""
    k = 2 * np.pi * np.fft.fftfreq(n)
    decay = np.exp(-(k[:, None] ** 2 + k[None, :] ** 2) / 4)
    u = [rng.normal(size=(n_batch, n, n))]
    for _ in range(n_time - 1):
        u.append(np.real(np.fft.ifft2(decay * np.fft.fft2(u[-1]))))
    return Field(values=np.stack(u, axis=1), scale=1.0, batched=True, time=np.arange(n_time))


def _fourier_arrays(d):
    """(residual, memory) in Fourier space as arrays (batch, n_time - 2, n_features), one feature per mode."""
    return tuple(f.to_fourier().values.reshape(*f.lead_shape, -1) for f in (d.residual, d.memory))


@pytest.mark.parametrize("odd_grid, grid", [(True, [3, 1]), (False, [4, 2])])
def test_levels_shapes_and_scales(odd_grid, grid):
    rng = np.random.default_rng(1)
    data = mori_zwanzig_data(_random_trajectories(rng), n_levels=2, odd_grid=odd_grid)

    assert [d.level for d in data] == [1, 2]
    assert [d.block_size for d in data] == [2, 4]
    assert [d.scale for d in data] == [(8 / m, 8 / m) for m in grid]
    # the cutoff does not depend on the grid the kept modes are stored on
    np.testing.assert_allclose([d.k_max for d in data], [(np.pi / 2, np.pi / 2), (np.pi / 4, np.pi / 4)])
    for d, m in zip(data, grid):
        assert d.residual.shape == d.memory.shape == (200, 3, m, m)  # n_time - 2 time steps on the coarse grid
        assert d.residual.domain == d.memory.domain == "real"
        residual, memory = _fourier_arrays(d)
        assert residual.shape == memory.shape == (200, 3, m * m)


def test_odd_grid_has_no_empty_fourier_features():
    rng = np.random.default_rng(11)
    field = _random_trajectories(rng)
    odd = mori_zwanzig_data(field, n_levels=1)[0]
    even = mori_zwanzig_data(field, n_levels=1, odd_grid=False)[0]

    for d, n_features, n_zero in [(odd, 9, 0), (even, 16, 7)]:
        for features in _fourier_arrays(d):
            features = features.reshape(-1, features.shape[-1])
            assert features.shape[1] == n_features
            assert np.sum(np.abs(features).max(axis=0) < 1e-12) == n_zero
            assert np.linalg.matrix_rank(features) == 9

    # the same modes, so the same data up to the empty Nyquist slots
    for a, b in zip(_fourier_arrays(odd), _fourier_arrays(even)):
        b = b.reshape(*b.shape[:2], 4, 4)[:, :, [0, 1, 3]][:, :, :, [0, 1, 3]]
        np.testing.assert_allclose(a, b.reshape(a.shape), atol=1e-12)


def test_markovian_dynamics_have_no_residual_or_memory():
    rng = np.random.default_rng(2)
    for d in mori_zwanzig_data(_linear_trajectories(rng), n_levels=2):
        np.testing.assert_allclose(d.residual.values, 0, atol=1e-10)
        np.testing.assert_allclose(d.memory.values, 0, atol=1e-10)


def test_memory_lies_in_the_resolved_span():
    rng = np.random.default_rng(3)
    field = _random_trajectories(rng)
    d = mori_zwanzig_data(field, n_levels=1)[0]
    assert np.abs(d.memory.values).max() > 1e-3

    # P (memory) = memory at every t: P is idempotent, with fixed coefficients from t = 0 evaluated at u[t]
    coarse = BlockCutoffFilter(block_size=2, odd_grid=True).transform(field)
    projection = Projection(BlockCutoffFilter(block_size=1), coarse, homogeneous=True)
    np.testing.assert_allclose(projection.evaluate(d.memory).values, d.memory.values, atol=1e-10)


def test_projection_uses_only_the_initial_states():
    rng = np.random.default_rng(12)
    field = _random_trajectories(rng)
    d = mori_zwanzig_data(field, n_levels=1)[0]

    # P regresses on u[:, 0] only, so the residual at t = 0 is u[1] - u[0] minus its best linear estimate from u[0]
    u = BlockCutoffFilter(block_size=2, odd_grid=True).transform(field)
    assert u.time is not None
    P = Projection(BlockCutoffFilter(block_size=1), u.replace(values=u.values[:, :1], time=u.time[:1]), homogeneous=True)
    Lu0 = u.replace(values=u.values[:, 1:2] - u.values[:, :1], time=u.time[:1])
    np.testing.assert_allclose(d.residual.values[:, :1], Lu0.values - P.evaluate(Lu0).values, atol=1e-12)


def test_too_few_samples_warns():
    rng = np.random.default_rng(4)
    with pytest.warns(UserWarning, match="rank deficient"):
        mori_zwanzig_data(_random_trajectories(rng, n_batch=2), n_levels=1, homogeneous=False)


def test_homogeneous_projection_needs_far_fewer_samples():
    rng = np.random.default_rng(9)
    field = _random_trajectories(rng, n_batch=2)  # 2 initial states: fewer than the 9 modes, enough for 1 component
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        d = mori_zwanzig_data(field, n_levels=1)[0]
    assert np.abs(d.residual.values).max() > 1e-3


def test_homogeneous_and_full_projection_agree_on_a_translation_invariant_ensemble():
    rng = np.random.default_rng(10)
    values = rng.normal(size=(4, 4, 8, 8))
    # every periodic translation of every trajectory, so cross-wavenumber correlations vanish exactly
    values = np.concatenate([np.roll(values, (i, j), axis=(2, 3)) for i in range(8) for j in range(8)])
    field = Field(values=values, scale=1.0, batched=True, time=np.arange(4))
    homogeneous = mori_zwanzig_data(field, n_levels=2)
    full = mori_zwanzig_data(field, n_levels=2, homogeneous=False)
    for h, f in zip(homogeneous, full):
        np.testing.assert_allclose(h.residual.values, f.residual.values, atol=1e-8)
        np.testing.assert_allclose(h.memory.values, f.memory.values, atol=1e-8)


def test_requires_batched_time_series():
    rng = np.random.default_rng(5)
    with pytest.raises(ValueError, match="time axis"):
        mori_zwanzig_data(Field(values=rng.normal(size=(4, 8, 8)), scale=1.0, batched=True), n_levels=1)
    with pytest.raises(ValueError, match="not enough time steps"):
        mori_zwanzig_data(_random_trajectories(rng, n_time=2), n_levels=1)


def _with(d, residual=None, memory=None):
    """d with its residual and memory values replaced."""
    return dataclasses.replace(
        d,
        residual=d.residual if residual is None else d.residual.replace(values=residual),
        memory=d.memory if memory is None else d.memory.replace(values=memory),
    )


def test_ratio_is_the_least_squares_fit_over_all_batch_members():
    rng = np.random.default_rng(13)
    d = mori_zwanzig_data(_random_trajectories(rng, n_time=6), n_levels=1)[0]
    gamma, stderr = d.ratio()
    assert gamma.shape == stderr.shape == (4,)
    assert not np.iscomplexobj(gamma)
    for t in range(4):
        m = d.memory.values[:, t].reshape(-1, 1)
        r = d.residual.values[:, t].reshape(-1)
        np.testing.assert_allclose(gamma[t], np.linalg.lstsq(m, r, rcond=None)[0][0], rtol=1e-10, atol=1e-12)
    # the same fit in Fourier space (Parseval)
    r, m = _fourier_arrays(d)
    np.testing.assert_allclose(np.real(np.sum(m.conj() * r, axis=(0, 2))) / np.sum(np.abs(m) ** 2, axis=(0, 2)), gamma, rtol=1e-10, atol=1e-12)


def test_ratio_vanishes_at_the_initial_time():
    # at t = 0, Q L u is orthogonal to the span of P, which contains P L Q L u
    rng = np.random.default_rng(14)
    for homogeneous in (True, False):
        for d in mori_zwanzig_data(_random_trajectories(rng), n_levels=2, homogeneous=homogeneous):
            gamma, _ = d.ratio()
            np.testing.assert_allclose(gamma[0], 0, atol=1e-10)
            assert np.all(np.abs(gamma[1:]) > 1e-6)


def test_ratio_of_exactly_proportional_data_has_no_error():
    rng = np.random.default_rng(15)
    d = mori_zwanzig_data(_random_trajectories(rng), n_levels=1)[0]
    factors = np.array([0.5, -2.0, 3.0])
    d = _with(d, residual=factors[None, :, None, None] * d.memory.values)
    gamma, stderr = d.ratio()
    np.testing.assert_allclose(gamma, factors, rtol=1e-12)
    np.testing.assert_allclose(stderr, 0, atol=1e-12)
    np.testing.assert_allclose(d.noise().values, 0, atol=1e-12)
    np.testing.assert_allclose(noise_data([d]).noise_var, 0, atol=1e-24)


def test_noise_is_centered_and_uncorrelated_with_the_memory():
    rng = np.random.default_rng(18)
    d = mori_zwanzig_data(_random_trajectories(rng, n_time=6), n_levels=1)[0]
    gamma, _ = d.ratio()
    noise = d.noise()
    np.testing.assert_allclose(noise.values + gamma[None, :, None, None] * d.memory.values, d.residual.values)
    # the least-squares noise is orthogonal to the memory at every t, over all batch members
    np.testing.assert_allclose(np.sum(noise.values * d.memory.values, axis=(0, 2, 3)), 0, atol=1e-10)
    # and centered: its batch mean is within sampling error of zero
    z = noise.values.mean(axis=0) / (noise.values.std(axis=0) / np.sqrt(noise.shape[0]))
    assert np.abs(z).max() < 5


def test_ratio_is_nan_without_memory():
    rng = np.random.default_rng(16)
    d = mori_zwanzig_data(_linear_trajectories(rng), n_levels=1)[0]
    d = _with(d, memory=np.zeros_like(d.memory.values))
    assert np.all(np.isnan(d.ratio()[0]))


def test_ratio_data_has_one_row_per_level_and_time(tmp_path):
    rng = np.random.default_rng(17)
    data = mori_zwanzig_data(_random_trajectories(rng, n_time=6), n_levels=2)
    rows = ratio_data(data, n_bootstrap=50, seed=0)

    assert rows.time.shape == rows.level.shape == rows.block_size.shape == rows.ratio.shape == (8,)
    assert rows.stderr.shape == (8,)
    assert rows.k_max.shape == (8, 2) and rows.bootstrap.shape == (8, 50)
    np.testing.assert_array_equal(rows.time, np.tile(np.arange(4), 2))
    np.testing.assert_array_equal(rows.level, [1] * 4 + [2] * 4)
    np.testing.assert_array_equal(rows.block_size, [2] * 4 + [4] * 4)
    np.testing.assert_allclose(rows.k_max[:, 0], [np.pi / 2] * 4 + [np.pi / 4] * 4)
    np.testing.assert_allclose(rows.ratio, np.concatenate([d.ratio()[0] for d in data]))

    # the bootstrap spread matches the standard error away from t = 0, and the seed makes it reproducible
    later = rows.time > 0
    np.testing.assert_allclose(rows.bootstrap[later].std(axis=1), rows.stderr[later], rtol=0.5)
    np.testing.assert_array_equal(ratio_data(data, n_bootstrap=50, seed=0).bootstrap, rows.bootstrap)

    rows.save(tmp_path / "ratio.npz")
    loaded = RatioData(**np.load(tmp_path / "ratio.npz"))
    for name in RatioData.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(loaded, name), getattr(rows, name))


@pytest.mark.parametrize("n_components", [0, 1])
def test_noise_data_has_one_row_per_level_time_mode_and_component(n_components, tmp_path):
    rng = np.random.default_rng(19)
    shape = (200, 5, 8, 8) if n_components == 0 else (200, 5, 8, 8, 2)
    field = Field(values=rng.normal(size=shape), scale=1.0, batched=True, time=np.arange(5), n_components=n_components)
    data = mori_zwanzig_data(field, n_levels=2)
    rows = noise_data(data)

    n_c = 1 if n_components == 0 else 2
    n_rows = 3 * (9 + 1) * n_c  # 3 time steps; 3x3 modes at level 1, 1 mode at level 2
    assert rows.time.shape == rows.level.shape == rows.component.shape == rows.noise_var.shape == (n_rows,)
    assert rows.k.shape == (n_rows, 2)

    level1 = rows.level == 1
    k = 2 * np.pi / 8 * np.array([0, 1, -1])  # the kept wavenumbers of the 8-site axis, in fftfreq order
    expected_k = np.stack(np.meshgrid(k, k, indexing="ij"), axis=-1).reshape(-1, 2)
    np.testing.assert_allclose(rows.k[level1 & (rows.time == 0) & (rows.component == 0)], expected_k)
    np.testing.assert_allclose(rows.k[rows.level == 2], 0)

    # the variances of a mode, its row by row value, and their sum over modes (Parseval)
    noise = data[0].noise()
    var = np.mean(np.abs(noise.to_fourier().values) ** 2, axis=0)  # (time, 3, 3, *components)
    np.testing.assert_allclose(rows.noise_var[level1], var.ravel())
    per_point = np.mean(noise.values ** 2, axis=(0, 2, 3))
    per_point = per_point if n_components == 0 else per_point.sum(axis=-1)
    total = [rows.noise_var[level1 & (rows.time == t)].sum() for t in range(3)]
    np.testing.assert_allclose(total, per_point)

    rows.save(tmp_path / "noise.npz")
    loaded = NoiseData(**np.load(tmp_path / "noise.npz"))
    for name in NoiseData.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(loaded, name), getattr(rows, name))
