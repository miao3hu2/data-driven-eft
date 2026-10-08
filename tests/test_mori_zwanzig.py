import numpy as np
import pytest

from core.coarse_graining.filter import BlockCutoffFilter
from core.data.base import Field
from core.operators.base import Operator
from core.operators.ising import NearestNeighbor
from core.operators.mori_zwanzig import Projection

X = np.array(
    [
        [1, 1, -1, 1],
        [1, -1, -1, -1],
        [-1, -1, 1, 1],
        [1, -1, -1, 1],
    ]
)


KEPT = [0, 1, 7]  # |k| < k_max on an 8-site axis with block_size 2


def _random_field(rng, n_batch=500, n_time=None, n=8):
    shape = (n_batch, n, n) if n_time is None else (n_batch, n_time, n, n)
    time = None if n_time is None else np.arange(n_time)
    return Field(values=rng.normal(size=shape), scale=1.0, batched=True, time=time)


def _kept_modes(values):
    """Kept Fourier modes on the last two (spatial) axes."""
    return values[..., KEPT, :][..., KEPT]


def test_basis_and_gram():
    rng = np.random.default_rng(0)
    field = _random_field(rng)
    projection = Projection(BlockCutoffFilter(block_size=2), field)

    a = projection.basis()
    assert a.shape == (500, 9)  # |k| < k_max keeps 3x3 modes of an 8x8 grid
    np.testing.assert_allclose(a, _kept_modes(field.to_fourier().values).reshape(500, 9))
    np.testing.assert_allclose(projection.gram, projection.gram.conj().T)


def test_projection_of_the_field_itself():
    rng = np.random.default_rng(3)
    field = _random_field(rng)
    projection = Projection(BlockCutoffFilter(block_size=2), field)

    # the kept Fourier modes of O = x are exactly the basis: coefficient 1 on its own mode, 0 elsewhere
    c = projection.coefficients(field)
    assert c.shape == (9, 8, 8)  # (n_modes, *spatial Fourier modes of O)
    np.testing.assert_allclose(_kept_modes(c).reshape(9, 9), np.eye(9), atol=1e-10)

    # a real field gives a real Field back on the real grid
    projected = projection.evaluate(field)
    assert isinstance(projected, Field)
    assert projected.domain == "real" and projected.batched and projected.shape == (500, 8, 8)
    assert not np.iscomplexobj(projected.values)
    # its kept modes are those of x itself (the high modes only pick up sample correlations with the basis)
    np.testing.assert_allclose(
        _kept_modes(projected.to_fourier().values), _kept_modes(field.to_fourier().values), atol=1e-10
    )

    # a Fourier field gives the same projection in Fourier space
    projected_hat = projection.evaluate(field.to_fourier())
    assert projected_hat.domain == "fourier"
    np.testing.assert_allclose(projected_hat.values, projected.to_fourier().values, atol=1e-10)


def test_projection_residual_is_orthogonal_to_basis():
    rng = np.random.default_rng(1)
    field = _random_field(rng)
    projection = Projection(BlockCutoffFilter(block_size=2), field)

    op_field = NearestNeighbor().evaluate(field)
    residual = op_field.to_fourier().values - projection.evaluate(op_field).to_fourier().values
    a = projection.basis()
    np.testing.assert_allclose(np.einsum("bk,bij->kij", a.conj(), residual) / 500, 0, atol=1e-10)


def test_projection_uses_time_zero_basis():
    rng = np.random.default_rng(2)
    field = _random_field(rng, n_time=3)
    projection = Projection(BlockCutoffFilter(block_size=2), field)

    t0 = field.replace(values=field.values[:, 0], time=None)
    np.testing.assert_allclose(projection.basis(), Projection(BlockCutoffFilter(block_size=2), t0).basis())

    c = projection.coefficients(field)
    assert c.shape == (9, 3, 8, 8)  # (n_modes, n_time, *spatial Fourier modes)
    np.testing.assert_allclose(_kept_modes(c[:, 0]).reshape(9, 9), np.eye(9), atol=1e-10)  # x(0) lies in the span

    projected = projection.evaluate(field)
    assert projected.shape == (500, 3, 8, 8)
    np.testing.assert_array_equal(projected.time, field.time)


def test_projection_is_real_only_for_real_field_and_operator():
    rng = np.random.default_rng(5)
    field = _random_field(rng)
    projection = Projection(BlockCutoffFilter(block_size=2), field)

    op_field = field.replace(values=field.values ** 2)
    real = projection.evaluate(op_field).values
    assert not np.iscomplexobj(real)

    # complex operator: P is linear, so the result is (1 + i) times the real one and stays complex
    complex_ = projection.evaluate(op_field.replace(values=(1 + 1j) * op_field.values)).values
    assert np.iscomplexobj(complex_)
    np.testing.assert_allclose(complex_, (1 + 1j) * real, atol=1e-10)


def test_projection_is_an_operator():
    rng = np.random.default_rng(6)
    field = _random_field(rng)
    projection = Projection(BlockCutoffFilter(block_size=2), field)
    assert isinstance(projection, Operator)

    # P x = x up to the unresolved modes, so P P x = P x
    once = projection.evaluate(field)
    np.testing.assert_allclose(projection.evaluate(once).values, once.values, atol=1e-10)


def test_op_field_must_match_the_ensemble_batch():
    rng = np.random.default_rng(4)
    projection = Projection(BlockCutoffFilter(block_size=2), _random_field(rng))
    with pytest.raises(ValueError, match="batched"):
        projection.evaluate(_random_field(rng, n_batch=7))
    with pytest.raises(ValueError, match="batched"):
        projection.evaluate(Field(values=X, scale=1.0))


def test_unbatched_ensemble_is_rejected():
    with pytest.raises(ValueError, match="batched"):
        Projection(BlockCutoffFilter(block_size=2), Field(values=X, scale=1.0))


def _translations(values):
    """Every periodic translation of every sample on the two spatial axes after the batch axis: an ensemble whose
    cross-wavenumber correlations vanish exactly."""
    n1, n2 = values.shape[1:3]
    shifted = [np.roll(values, (i, j), axis=(1, 2)) for i in range(n1) for j in range(n2)]
    return np.concatenate(shifted)


@pytest.mark.parametrize("n_components", [0, 1])
def test_homogeneous_projection_matches_full_projection_on_a_translation_invariant_ensemble(n_components):
    rng = np.random.default_rng(7)
    shape = (20, 8, 8) if n_components == 0 else (20, 8, 8, 2)
    ensemble = Field(values=_translations(rng.normal(size=shape)), scale=1.0, batched=True, n_components=n_components)
    # a translation-equivariant nonlinear operator, so its values are translated with the ensemble
    op_field = ensemble.replace(values=ensemble.values ** 2 + np.roll(ensemble.values, 1, axis=1))

    full = Projection(BlockCutoffFilter(block_size=2), ensemble)
    homogeneous = Projection(BlockCutoffFilter(block_size=2), ensemble, homogeneous=True)

    n_c = 1 if n_components == 0 else 2
    assert homogeneous.gram.shape == (9, n_c, n_c)  # one block per kept wavenumber
    np.testing.assert_allclose(homogeneous.evaluate(op_field).values, full.evaluate(op_field).values, atol=1e-10)


def test_homogeneous_projection_handles_singular_blocks():
    # the second component copies the first, so every 2x2 block has rank 1 (like a_k orthogonal to k for
    # divergence-free velocity)
    rng = np.random.default_rng(8)
    x = _translations(rng.normal(size=(20, 8, 8)))
    ensemble = Field(values=np.stack([x, x], axis=-1), scale=1.0, batched=True, n_components=1)
    op_field = ensemble.replace(values=ensemble.values ** 2)

    full = Projection(BlockCutoffFilter(block_size=2), ensemble)
    homogeneous = Projection(BlockCutoffFilter(block_size=2), ensemble, homogeneous=True)
    np.testing.assert_allclose(np.linalg.matrix_rank(homogeneous.gram, hermitian=True), 1)
    np.testing.assert_allclose(homogeneous.evaluate(op_field).values, full.evaluate(op_field).values, atol=1e-10)


def test_homogeneous_projection_keeps_time_and_fourier_layout():
    rng = np.random.default_rng(9)
    field = _random_field(rng, n_time=3)
    homogeneous = Projection(BlockCutoffFilter(block_size=2), field, homogeneous=True)

    c = homogeneous.coefficients(field)
    assert c.shape == (9, 1, 3)  # (n_wavenumbers, n_components, n_time)
    np.testing.assert_allclose(c[:, 0, 0], 1, atol=1e-10)  # x(0) is the basis itself

    projected = homogeneous.evaluate(field.to_fourier())
    assert projected.domain == "fourier" and projected.shape == (500, 3, 8, 8)
    # modes outside the kept set are uncorrelated with the basis, so P sets them to zero
    outside = np.ones((8, 8), dtype=bool)
    outside[np.ix_(KEPT, KEPT)] = False
    np.testing.assert_array_equal(projected.values[:, :, outside], 0)
    np.testing.assert_allclose(
        _kept_modes(projected.values[:, 0]), _kept_modes(field.to_fourier().values[:, 0]), atol=1e-10
    )


def test_homogeneous_projection_needs_the_ensemble_grid():
    rng = np.random.default_rng(10)
    field = _random_field(rng)
    homogeneous = Projection(BlockCutoffFilter(block_size=2), field, homogeneous=True)
    coarse = BlockCutoffFilter(block_size=2).transform(field)
    with pytest.raises(ValueError, match="grid"):
        homogeneous.evaluate(coarse)
