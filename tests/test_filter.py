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


def test_odd_grid_stores_one_point_per_kept_mode():
    L = 16
    low = lambda x, y: np.cos(2 * np.pi / L * x) + 0.5 * np.cos(2 * np.pi / L * (2 * x + 3 * y))
    high = _plane_waves(L, [(0.7, 6, 0), (0.3, 4, 1)])  # |mode| >= L / (2 * 2) = 4, removed (4 is the coarse Nyquist)
    x = np.arange(L)
    field = Field(values=low(*np.meshgrid(x, x, indexing="ij")) + high, scale=1.0)

    coarse = BlockCutoffFilter(block_size=2, odd_grid=True).transform(field)

    assert coarse.values.shape == (7, 7)
    assert coarse.scale == (16 / 7, 16 / 7)
    assert not np.iscomplexobj(coarse.values)
    # the low-mode signal sampled on the odd grid, whose points are not fine grid points
    xc = np.arange(7) * 16 / 7
    np.testing.assert_allclose(coarse.values, low(*np.meshgrid(xc, xc, indexing="ij")), atol=1e-12)


def test_odd_grid_holds_the_same_modes_without_the_nyquist_slot():
    field = Field(values=np.random.default_rng(2).normal(size=(3, 12, 8)), scale=1.0, batched=True).to_fourier()
    odd = BlockCutoffFilter(block_size=2, odd_grid=True).transform(field)
    even = BlockCutoffFilter(block_size=2).transform(field)

    assert odd.values.shape == (3, 5, 3)  # coarse sizes 6 and 4 lose their Nyquist slot
    np.testing.assert_array_equal(odd.values, even.values[:, [0, 1, 2, 4, 5]][:, :, [0, 1, 3]])
    np.testing.assert_array_equal(np.delete(even.values, 3, axis=1)[:, :, 2], 0)
    # project and basis only depend on the kept modes, not on the grid that stores them
    np.testing.assert_array_equal(
        BlockCutoffFilter(block_size=2, odd_grid=True).project(field).values, BlockCutoffFilter(block_size=2).project(field).values
    )


def test_odd_grid_keeps_every_mode_of_an_odd_grid():
    field = Field(values=np.random.default_rng(3).normal(size=(7, 7)), scale=1.0)
    np.testing.assert_allclose(BlockCutoffFilter(block_size=1, odd_grid=True).transform(field).values, field.values)
    np.testing.assert_allclose(BlockCutoffFilter(block_size=1).transform(field).values, field.values)


def test_odd_grid_transforms_compose_on_power_of_two_grids():
    field = Field(values=np.random.default_rng(4).normal(size=(32, 32)), scale=1.0)
    f = BlockCutoffFilter(block_size=2, odd_grid=True)
    twice = f.transform(f.transform(field))
    once = BlockCutoffFilter(block_size=4, odd_grid=True).transform(field)
    assert twice.values.shape == once.values.shape == (7, 7)
    np.testing.assert_allclose(twice.values, once.values, atol=1e-12)
    np.testing.assert_allclose(twice.scale, once.scale)


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


@pytest.mark.parametrize("n, block_size", SIZES)
@pytest.mark.parametrize("domain", ["real", "fourier"])
def test_basis_outer_product_is_the_projector(n, block_size, domain):
    # B B^H, contracted over the mode (batch) axis, is the projector P[x, y, a, b] onto the kept modes
    f = BlockCutoffFilter(block_size=block_size)
    shape = (n, n + block_size)
    field = Field(values=np.random.default_rng(8).normal(size=shape), scale=(1.0, 0.5))
    if domain == "fourier":
        field = field.to_fourier()

    B = f.basis(field).values
    N = shape[0] * shape[1]
    P = np.einsum("kxy,kab->xyab", B, B.conj())
    if domain == "real":
        P /= N
    M = P.reshape(N, N)

    np.testing.assert_allclose(np.einsum("xyab,ab->xy", P, field.values), f.project(field).values, atol=1e-12)
    np.testing.assert_allclose(M @ M, M, atol=1e-12)  # idempotent
    np.testing.assert_allclose(M, M.conj().T, atol=1e-12)  # Hermitian
    np.testing.assert_allclose(np.trace(M), B.shape[0], atol=1e-9)  # rank = n_kept

    if domain == "real":
        # a real convolution kernel: P[x, y, a, b] depends only on (x - a, y - b)
        assert np.abs(M.imag).max() < 1e-12
        np.testing.assert_allclose(np.roll(P, (3, 5, 3, 5), axis=(0, 1, 2, 3)), P, atol=1e-12)
    else:
        # the cutoff mask: diagonal with 1 at the kept modes and 0 elsewhere
        np.testing.assert_allclose(M, np.diag(np.diag(M)), atol=1e-12)
        assert set(np.unique(np.diag(M))) <= {0, 1}


@pytest.mark.parametrize("n, block_size", SIZES)
@pytest.mark.parametrize("domain", ["real", "fourier"])
def test_inner_products_with_basis_are_the_kept_fourier_coefficients(n, block_size, domain):
    f = BlockCutoffFilter(block_size=block_size)
    shape = (n, n + block_size)
    field = Field(values=np.random.default_rng(9).normal(size=shape), scale=(1.0, 0.5))
    fourier = field.to_fourier()
    if domain == "fourier":
        field = fourier

    # <phi_k, u>, with the 1 / N of the "forward" normalization in real space
    coeffs = np.einsum("kxy,xy->k", f.basis(field).values.conj(), field.values)
    if domain == "real":
        coeffs /= shape[0] * shape[1]

    m = [ni // block_size for ni in shape]
    fine = [np.r_[0:(mi + 1) // 2, ni - (mi - 1) // 2:ni] for ni, mi in zip(shape, m)]
    coarse = [np.r_[0:(mi + 1) // 2, mi - (mi - 1) // 2:mi] for mi in m]
    coeffs = coeffs.reshape(len(fine[0]), len(fine[1]))
    np.testing.assert_allclose(coeffs, fourier.values[np.ix_(*fine)], atol=1e-12)
    # the same coefficients sit at the kept modes of the coarse grid, with no rescaling
    coarse_fourier = f.transform(field).to_fourier().values
    np.testing.assert_allclose(coeffs, coarse_fourier[np.ix_(*coarse)], atol=1e-12)
