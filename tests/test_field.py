import dataclasses
import numpy as np
import pytest

from core.data.base import Field, Grid


def test_field_scalar_scale_broadcasts_to_all_spatial_dims():
    field = Field(values=np.zeros((3, 4)), scale=1.5)
    assert field.scale == (1.5, 1.5)


def test_field_single_element_scale_broadcasts_to_all_spatial_dims():
    field = Field(values=np.zeros((3, 4)), scale=[2.0])
    assert field.scale == (2.0, 2.0)


def test_field_sequence_scale_kept_as_given():
    field = Field(values=np.zeros((3, 4)), scale=(1.0, 2.0))
    assert field.scale == (1.0, 2.0)


def test_field_mismatched_scale_length_raises():
    with pytest.raises(ValueError):
        Field(values=np.zeros((3, 4)), scale=(1.0, 2.0, 3.0))


def test_field_batched_scale_excludes_batch_dim():
    field = Field(values=np.zeros((5, 3, 4)), scale=1.0, batched=True)
    assert field.scale == (1.0, 1.0)


def test_field_default_metadata_is_empty_dict():
    field = Field(values=np.zeros((2, 2)), scale=1.0)
    assert field.metadata == {}


def test_field_shape_property_matches_values_shape():
    values = np.zeros((3, 4))
    field = Field(values=values, scale=1.0)
    assert field.shape == (3, 4)


def test_field_values_coerced_to_ndarray():
    field = Field(values=[[1, 2], [3, 4]], scale=1.0)
    assert isinstance(field.values, np.ndarray)
    np.testing.assert_array_equal(field.values, [[1, 2], [3, 4]])


def test_grid_coordinates_shape_and_values():
    grid = Grid(shape=(3, 2), spacing=(1.0, 2.0))
    x, y = grid.coordinates

    assert x.shape == (3, 2)
    assert y.shape == (3, 2)

    np.testing.assert_array_equal(x[:, 0], [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(y[0, :], [0.0, 2.0])


def test_field_is_immutable():
    field = Field(values=np.zeros((2, 2)), scale=1.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        field.scale = (2.0, 2.0)  # pyright: ignore[reportAttributeAccessIssue]


def test_field_layout_batch_time_space_components():
    # (batch, time, x, y, z, component) e.g. a batch of velocity-field trajectories
    values = np.zeros((2, 5, 4, 6, 8, 3))
    field = Field(values=values, scale=0.5, batched=True, time=np.linspace(0, 1, 5), n_components=1)

    assert field.n_lead == 2
    assert field.time_axis == 1
    assert field.spatial_axes == (2, 3, 4)
    assert field.spatial_shape == (4, 6, 8)
    assert field.component_axes == (5,)
    assert field.component_shape == (3,)
    assert field.lead_shape == (2, 5)
    assert field.scale == (0.5, 0.5, 0.5)


def test_field_time_without_batch():
    field = Field(values=np.zeros((5, 4, 4)), scale=1.0, time=np.arange(5))
    assert field.time_axis == 0
    assert field.spatial_axes == (1, 2)


def test_field_time_length_mismatch_raises():
    with pytest.raises(ValueError):
        Field(values=np.zeros((5, 4, 4)), scale=1.0, time=np.arange(3))


def test_field_invalid_domain_raises():
    with pytest.raises(ValueError):
        Field(values=np.zeros((4, 4)), scale=1.0, domain="momentum")  # pyright: ignore[reportArgumentType]


def test_field_replace_carries_other_attributes():
    field = Field(values=np.zeros((2, 3, 4)), scale=1.0, time=[0.0, 0.1], metadata={"a": 1})
    new = field.replace(values=np.ones((2, 3, 4)))

    np.testing.assert_array_equal(new.time, field.time)
    assert new.scale == field.scale
    assert new.metadata == {"a": 1}
    assert np.all(new.values == 1)


def test_field_fourier_round_trip_only_touches_spatial_axes():
    rng = np.random.default_rng(0)
    values = rng.normal(size=(3, 4, 8, 8, 2))
    field = Field(values=values, scale=1.0, batched=True, time=np.arange(4), n_components=1)

    fourier = field.to_fourier()
    assert fourier.domain == "fourier"
    np.testing.assert_allclose(fourier.values, np.fft.fftn(values, axes=(2, 3), norm="forward"))

    back = fourier.to_real()
    assert back.domain == "real"
    np.testing.assert_allclose(back.values, values)


def test_field_wavenumbers():
    field = Field(values=np.zeros((4, 8)), scale=(0.5, 1.0))
    kx, ky = field.wavenumbers()
    np.testing.assert_allclose(kx, 2 * np.pi * np.fft.fftfreq(4, d=0.5))
    np.testing.assert_allclose(ky, 2 * np.pi * np.fft.fftfreq(8, d=1.0))


def test_fourier_round_trip_recovers_real_field():
    values = np.random.default_rng(0).normal(size=(2, 16, 12))
    field = Field(values=values, scale=0.5, batched=True)
    back = field.to_fourier().to_real(real=False)
    # exact up to floating-point roundoff, with a negligible imaginary part
    np.testing.assert_allclose(np.real(back.values), values, atol=1e-12)
    assert np.abs(np.imag(back.values)).max() < 1e-12
