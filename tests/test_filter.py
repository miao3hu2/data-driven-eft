import numpy as np
import pytest

from core.data.base import Field
from core.coarse_graining.filter import BlockCutoffFilter


def _plane_waves(L, modes, dx=1.0) -> np.ndarray:
    x = np.arange(L) * dx
    X, Y = np.meshgrid(x, x, indexing="ij")
    k0 = 2 * np.pi / (L * dx)
    out = np.zeros_like(X)
    for a, m, n in modes:
        out += a * np.cos(k0 * (m * X + n * Y))
    return out


def test_transform_keeps_low_modes_on_coarser_grid():
    L = 16
    low = _plane_waves(L, [(1.0, 1, 0), (0.5, 2, 3)])
    high = _plane_waves(L, [(0.7, 6, 0)])  # |mode| 6 > L / (2 * 2) = 4, removed
    field = Field(values=low + high, scale=1.0)

    coarse = BlockCutoffFilter(block_size=2).transform(field)

    assert coarse.values.shape == (8, 8)
    assert coarse.scale == (2.0, 2.0)
    assert coarse.domain == "real"
    assert not np.iscomplexobj(coarse.values)
    # the low-mode signal sampled on the coarse grid
    np.testing.assert_allclose(coarse.values, low[::2, ::2], atol=1e-12)


def test_project_zeroes_high_modes_on_same_grid():
    L = 16
    low = _plane_waves(L, [(1.0, 1, 2)])
    high = _plane_waves(L, [(0.7, 0, 7)])
    field = Field(values=low + high, scale=1.0)

    projected = BlockCutoffFilter(block_size=2).project(field)

    assert projected.values.shape == (16, 16)
    assert projected.scale == (1.0, 1.0)
    np.testing.assert_allclose(projected.values, low, atol=1e-12)


def test_transform_preserves_batch_time_and_component_axes():
    rng = np.random.default_rng(0)
    values = rng.normal(size=(2, 3, 8, 8, 2))
    field = Field(values=values, scale=1.0, batched=True, time=[0.0, 0.1, 0.2], n_components=1)

    coarse = BlockCutoffFilter(block_size=2).transform(field)

    assert coarse.values.shape == (2, 3, 4, 4, 2)
    np.testing.assert_array_equal(coarse.time, field.time)
    assert coarse.n_components == 1
    # the k = 0 mode (spatial mean) is untouched by coarse-graining
    np.testing.assert_allclose(coarse.values.mean(axis=(2, 3)), values.mean(axis=(2, 3)))


def test_fourier_input_stays_in_fourier_domain():
    field = Field(values=np.random.default_rng(1).normal(size=(8, 8)), scale=1.0).to_fourier()
    coarse = BlockCutoffFilter(block_size=2).transform(field)
    assert coarse.domain == "fourier"
    assert coarse.values.shape == (4, 4)


def test_block_size_larger_than_axis_raises():
    with pytest.raises(ValueError):
        BlockCutoffFilter(block_size=8).transform(Field(values=np.zeros((4, 4)), scale=1.0))


# (n, block_size) pairs giving an even and an odd number of kept modes m = n // block_size
SIZES = [(16, 2), (16, 4), (12, 4), (15, 3)]


@pytest.mark.parametrize("n, block_size", SIZES)
@pytest.mark.parametrize("method", ["transform", "project"])
def test_filtered_spectrum_is_conjugate_symmetric(n, block_size, method):
    # a real field must stay real: the imaginary part dropped by to_real is only roundoff
    field = Field(values=np.random.default_rng(2).normal(size=(n, n)), scale=1.0).to_fourier()
    filtered = getattr(BlockCutoffFilter(block_size=block_size), method)(field)
    values = filtered.to_real(real=False).values
    assert np.abs(values.imag).max() < 1e-12


@pytest.mark.parametrize("n, block_size", SIZES)
def test_project_is_idempotent(n, block_size):
    f = BlockCutoffFilter(block_size=block_size)
    field = Field(values=np.random.default_rng(3).normal(size=(n, n)), scale=1.0)
    once = f.project(field)
    np.testing.assert_allclose(f.project(once).values, once.values, atol=1e-12)


@pytest.mark.parametrize("n, block_size", [(16, 2), (16, 4)])
def test_transform_matches_project_sampled_on_coarse_grid(n, block_size):
    f = BlockCutoffFilter(block_size=block_size)
    field = Field(values=np.random.default_rng(4).normal(size=(n, n)), scale=1.0)
    coarse = f.transform(field)
    projected = f.project(field)
    np.testing.assert_allclose(coarse.values, projected.values[::block_size, ::block_size], atol=1e-12)


def test_nyquist_mode_of_coarse_grid_is_removed():
    L = 16
    low = _plane_waves(L, [(1.0, 3, 1)])
    nyquist = _plane_waves(L, [(0.7, 4, 0)])  # |mode| 4 = L / (2 * 2), the coarse Nyquist mode
    field = Field(values=low + nyquist, scale=1.0)

    f = BlockCutoffFilter(block_size=2)
    np.testing.assert_allclose(f.project(field).values, low, atol=1e-12)
    np.testing.assert_allclose(f.transform(field).values, low[::2, ::2], atol=1e-12)


@pytest.mark.parametrize("method", ["transform", "project"])
def test_fourier_input_is_not_mutated(method):
    field = Field(values=np.random.default_rng(2).normal(size=(2, 8, 8)), scale=1.0, batched=True).to_fourier()
    before = field.values.copy()
    getattr(BlockCutoffFilter(block_size=2), method)(field)
    np.testing.assert_array_equal(field.values, before)


@pytest.mark.parametrize("n, block_size", SIZES)
def test_basis_matches_inverse_fft_of_kept_modes(n, block_size):
    # each basis function is the real-space image of a unit coefficient at a kept mode
    field = Field(values=np.zeros((n, n + block_size)), scale=(1.0, 0.5))
    basis = BlockCutoffFilter(block_size=block_size).basis(field)

    # |k| < m / 2 along each axis: m modes for odd m, m - 1 for even m (the coarse Nyquist mode is dropped)
    m = (n // block_size, (n + block_size) // block_size)
    kept = [np.r_[0:(mi + 1) // 2, ni - (mi - 1) // 2:ni] for ni, mi in zip(field.shape, m)]
    assert basis.shape == (len(kept[0]) * len(kept[1]), n, n + block_size)
    assert basis.batched and basis.domain == "real" and basis.scale == field.scale

    for col, (i, j) in enumerate((i, j) for i in kept[0] for j in kept[1]):
        delta = np.zeros(field.shape, dtype=complex)
        delta[i, j] = 1.0
        expected = Field(values=delta, scale=field.scale, domain="fourier").to_real(real=False).values
        np.testing.assert_allclose(basis.values[col], expected, atol=1e-12)


@pytest.mark.parametrize("n, block_size", SIZES)
def test_projecting_onto_basis_matches_project(n, block_size):
    f = BlockCutoffFilter(block_size=block_size)
    field = Field(values=np.random.default_rng(5).normal(size=(n, n)), scale=1.0)

    B = f.basis(field).values.reshape(-1, n * n).T
    np.testing.assert_allclose(B.conj().T @ B, n * n * np.eye(B.shape[1]), atol=1e-9)

    projected = B @ (B.conj().T @ field.values.ravel()) / (n * n)
    np.testing.assert_allclose(projected.reshape(n, n), f.project(field).values, atol=1e-12)


@pytest.mark.parametrize("n, block_size", SIZES)
def test_basis_is_returned_in_the_domain_of_the_field(n, block_size):
    f = BlockCutoffFilter(block_size=block_size)
    field = Field(values=np.random.default_rng(6).normal(size=(n, n + block_size)), scale=(1.0, 0.5))

    real_basis = f.basis(field)
    fourier_basis = f.basis(field.to_fourier())
    assert real_basis.domain == "real"
    assert fourier_basis.domain == "fourier"

    # the Fourier-space basis is the transform of the real-space one: unit vectors at the kept modes
    np.testing.assert_allclose(fourier_basis.values, real_basis.to_fourier().values, atol=1e-12)
    np.testing.assert_allclose(fourier_basis.to_real(real=False).values, real_basis.values, atol=1e-12)
    assert set(np.unique(fourier_basis.values)) <= {0, 1}


@pytest.mark.parametrize("n, block_size", SIZES)
def test_projecting_onto_fourier_basis_matches_project(n, block_size):
    f = BlockCutoffFilter(block_size=block_size)
    field = Field(values=np.random.default_rng(7).normal(size=(n, n)), scale=1.0).to_fourier()

    B = f.basis(field).values.reshape(-1, n * n).T
    np.testing.assert_allclose(B.conj().T @ B, np.eye(B.shape[1]), atol=1e-12)

    projected = B @ (B.conj().T @ field.values.ravel())
    np.testing.assert_allclose(projected.reshape(n, n), f.project(field).values, atol=1e-12)
