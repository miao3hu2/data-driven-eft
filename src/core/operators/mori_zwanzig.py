import numpy as np
from core.coarse_graining.filter import BlockCutoffFilter
from core.data.base import Field
from core.operators.base import Operator, OperatorBasis


def kept_indices(filter: BlockCutoffFilter, field: Field) -> list[np.ndarray]:
    return [np.concatenate([np.arange(n)[s] for s, _ in pairs]) for _, n, _, pairs in filter._kept(field)]


def kept_modes(filter: BlockCutoffFilter, field: Field) -> np.ndarray:
    values = field.to_fourier().values
    for axis, idx in zip(field.spatial_axes, kept_indices(filter, field)):
        values = np.take(values, idx, axis=axis)
    return values.reshape(*field.lead_shape, -1)


def apply_coefficients(
    modes: np.ndarray,
    coeffs: np.ndarray,
    kept: list[np.ndarray],
    spatial_shape: tuple[int, ...],
    homogeneous: bool,
) -> np.ndarray:
    lead = modes.shape[:-1]
    if not homogeneous:
        return np.tensordot(modes, coeffs, axes=([-1], [0]))
    kept_shape = tuple(len(idx) for idx in kept)
    rest_shape = coeffs.shape[2:]
    a = modes.reshape(*lead, -1, coeffs.shape[1])
    projected = np.einsum("...ki,kir->...kr", a, coeffs.reshape(*coeffs.shape[:2], -1))
    full = np.zeros((*lead, *spatial_shape, *rest_shape), dtype=projected.dtype)
    full[(*[slice(None)] * len(lead), *np.ix_(*kept))] = projected.reshape(*lead, *kept_shape, *rest_shape)
    return full


class Projection(Operator):
    name = "P"

    def __init__(
        self,
        filter: BlockCutoffFilter,
        ensemble: Field,
        homogeneous: bool = False,
        operators: OperatorBasis | None = None,
    ):
        if not ensemble.batched:
            raise ValueError("Projection requires a batched ensemble to average over.")
        self.filter = filter
        self.homogeneous = homogeneous
        self.operators = operators
        self.features = ensemble if operators is None else operators.stack(ensemble)
        self._basis = self.basis() # (batch, n_modes), with n_modes = n_kept * n_components
        n_batch = self._basis.shape[0]
        if homogeneous:
            a = self._wavenumber_basis() # (batch, n_wavenumbers, n_components)
            self.gram = np.einsum("bki,bkj->kij", a.conj(), a) / n_batch # using homogeneity: different wavenumbers are uncorrelated, so G is block diagonal with one block per wavenumber
        else:
            self.gram = self._basis.conj().T @ self._basis / n_batch

    def basis(self) -> np.ndarray:
        field = self.features
        if field.has_time:
            field = field.replace(values=field.values[:, 0], time=None)
        return self._modes(field)

    def evaluate(self, field: Field) -> Field:
        initial, states = field, self.features
        if field.has_time:
            initial = field.replace(values=field.values[:, 0], time=None)
            states = states.replace(values=states.values[:, self._time_indices(field)], time=field.time)
        elif states.has_time:
            states = states.replace(values=states.values[:, 0], time=None)

        coeffs = self.coefficients(initial)
        projected = apply_coefficients(
            self._modes(states), coeffs, self._kept_indices(initial), initial.spatial_shape, self.homogeneous
        )
        result = field.to_fourier().replace(values=projected)
        if field.domain == "real":
            real = not np.iscomplexobj(field.values) and self._is_real(self.features)
            return result.to_real(real=real)
        return result

    def coefficients(self, op_field: Field) -> np.ndarray:
        n_batch = self._basis.shape[0]
        if not op_field.batched or op_field.shape[0] != n_batch:
            raise ValueError(f"op_field must be batched over the {n_batch} ensemble members, got shape {op_field.shape}.")
        if op_field.has_time:
            op_field = op_field.replace(values=op_field.values[:, 0], time=None)

        values = op_field.to_fourier().values
        if self.homogeneous:
            modes, rest_shape = self._wavenumber_modes(op_field, values)
            inner = np.einsum("bki,bkr->kir", self._wavenumber_basis().conj(), modes) / n_batch
            coeffs = np.linalg.pinv(self.gram, hermitian=True) @ inner
            return coeffs.reshape(*coeffs.shape[:2], *rest_shape)

        op_shape = values.shape[1:]
        inner = self._basis.conj().T @ values.reshape(n_batch, -1) / n_batch  # (n_modes, prod(op_shape))

        coeffs, *_ = np.linalg.lstsq(self.gram, inner, rcond=None)
        return coeffs.reshape(-1, *op_shape)

    def wavevectors(self) -> np.ndarray:
        """The kept wavevectors, in the order of the basis: (n_wavevectors, n_space)."""
        kept = [k[idx] for k, idx in zip(self.features.wavenumbers(), self._kept_indices(self.features))]
        return np.stack(np.meshgrid(*kept, indexing="ij"), axis=-1).reshape(-1, len(kept))

    def _time_indices(self, field: Field) -> np.ndarray:
        """Indices into the ensemble's time axis of the time steps of field, which must start at the ensemble's first."""
        times, field_times = self.features.time, field.time
        if times is None:
            raise ValueError("field has a time axis, but the ensemble has no trajectories to evaluate P O along.")
        assert field_times is not None
        idx = np.searchsorted(times, field_times)
        if (
            np.any(idx >= len(times))
            or not np.array_equal(times[idx], field_times)
            or field_times[0] != times[0]
        ):
            raise ValueError(f"the time steps of field must be time steps of the ensemble starting at t = {times[0]}.")
        return idx

    def _modes(self, field: Field) -> np.ndarray:
        return kept_modes(self.filter, field)

    def _kept_indices(self, field: Field) -> list[np.ndarray]:
        return kept_indices(self.filter, field)

    def _wavenumber_basis(self) -> np.ndarray:
        """The basis split into wavenumbers and components: (batch, n_wavenumbers, n_components)."""
        n_components = int(np.prod(self.features.component_shape))
        return self._basis.reshape(self._basis.shape[0], -1, n_components)

    def _wavenumber_modes(self, op_field: Field, values: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
        if op_field.spatial_shape != self.features.spatial_shape:
            raise ValueError(
                f"op_field must be on the ensemble's grid {self.features.spatial_shape}, got {op_field.spatial_shape}."
            )
        axes = op_field.spatial_axes
        for axis, idx in zip(axes, self._kept_indices(op_field)):
            values = np.take(values, idx, axis=axis)
        values = np.moveaxis(values, axes, range(1, 1 + len(axes)))  # (batch, *kept, *rest)
        kept_shape = values.shape[1:1 + len(axes)]
        rest_shape = values.shape[1 + len(axes):]
        return values.reshape(values.shape[0], int(np.prod(kept_shape)), -1), rest_shape

    @staticmethod
    def _is_real(field: Field) -> bool:
        return field.domain == "real" and not np.iscomplexobj(field.values)
