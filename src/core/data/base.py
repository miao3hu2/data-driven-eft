from __future__ import annotations
import dataclasses
from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike
from typing import Literal, Sequence

class Grid:
    def __init__(self, shape, spacing):
        self.shape = shape
        self.spacing = spacing
        self.coordinates = self._generate_coordinates()

    def _generate_coordinates(self):
        return np.meshgrid(*[np.arange(0, s) * sp for s, sp in zip(self.shape, self.spacing)], indexing='ij')

# Fourier normalization coefficients are stored with norm="forward". The coefficient of a mode is then independent of the grid size, so truncating modes (spectral coarse-graining) needs no rescaling.
FFT_NORM = "forward"

@dataclass(frozen=True, eq=False, init=False)
class Field:
    values: np.ndarray
    scale: tuple[float, ...]
    batched: bool
    metadata: dict
    time: np.ndarray | None
    n_components: int # Number of components in the field, 0 for scalar fields, 1 for vector fields, etc.
    domain: Literal["real", "fourier"]

    def __init__(
        self,
        values: ArrayLike,
        scale: float | Sequence[float],
        batched: bool = False,
        metadata: dict | None = None,
        time: ArrayLike | None = None,
        n_components: int = 0,
        domain: Literal["real", "fourier"] = "real",
    ):
        # frozen dataclass: set attributes through object.__setattr__
        set_ = lambda name, value: object.__setattr__(self, name, value)

        set_("values", np.asarray(values))
        set_("batched", bool(batched))
        set_("metadata", {} if metadata is None else metadata)
        set_("n_components", n_components)
        set_("domain", domain)
        set_("time", None if time is None else np.asarray(time, dtype=float))

        if self.domain not in ("real", "fourier"):
            raise ValueError(f"domain must be 'real' or 'fourier', got {self.domain!r}.")
        if self.n_components < 0:
            raise ValueError("n_components must be non-negative.")

        if self.time is not None:
            if self.time.ndim != 1:
                raise ValueError("time must be a 1D array of time coordinates.")
            if self.values.ndim <= self.n_batch or self.values.shape[self.n_batch] != len(self.time):
                raise ValueError("Length of time must match the time axis of values.")

        n_spatial_dims = self.n_space
        if n_spatial_dims < 1:
            raise ValueError("values must have at least one spatial dimension.")

        if isinstance(scale, (int, float, np.number)):
            scale_tuple = (float(scale),) * n_spatial_dims
        else:
            scale_tuple = tuple(float(s) for s in scale)
            if len(scale_tuple) == 1 and n_spatial_dims != 1: # in case scale is an array of a single number
                scale_tuple = scale_tuple * n_spatial_dims

        if len(scale_tuple) != n_spatial_dims:
            raise ValueError("Scale length must match the number of spatial dimensions in values.")
        set_("scale", scale_tuple)

    # --- axis layout ---

    @property
    def shape(self):
        return self.values.shape

    @property
    def n_batch(self) -> int:
        return int(self.batched)

    @property
    def has_time(self) -> bool:
        return self.time is not None

    @property
    def n_lead(self) -> int:
        """Number of leading (batch + time) axes."""
        return self.n_batch + int(self.has_time)

    @property
    def n_space(self) -> int:
        return self.values.ndim - self.n_lead - self.n_components

    @property
    def lead_axes(self) -> tuple[int, ...]:
        return tuple(range(self.n_lead))

    @property
    def time_axis(self) -> int | None:
        return self.n_batch if self.has_time else None

    @property
    def spatial_axes(self) -> tuple[int, ...]:
        return tuple(range(self.n_lead, self.n_lead + self.n_space))

    @property
    def component_axes(self) -> tuple[int, ...]:
        return tuple(range(self.n_lead + self.n_space, self.values.ndim))

    @property
    def lead_shape(self) -> tuple[int, ...]:
        return self.values.shape[:self.n_lead]

    @property
    def spatial_shape(self) -> tuple[int, ...]:
        return self.values.shape[self.n_lead:self.n_lead + self.n_space]

    @property
    def component_shape(self) -> tuple[int, ...]:
        return self.values.shape[self.n_lead + self.n_space:]

    # --- construction helpers ---

    def replace(self, **changes) -> Field:
        """Return a copy with the given attributes changed; everything else is carried over."""
        return dataclasses.replace(self, **changes)

    # --- Fourier space ---

    def wavenumbers(self) -> list[np.ndarray]:
        """Angular wavenumbers k_i (in np.fft.fftfreq order) along each spatial axis."""
        return [
            2 * np.pi * np.fft.fftfreq(n, d=dx)
            for n, dx in zip(self.spatial_shape, self.scale)
        ]

    def to_fourier(self) -> Field:
        if self.domain == "fourier":
            return self
        values = np.fft.fftn(self.values, axes=self.spatial_axes, norm=FFT_NORM)
        return self.replace(values=values, domain="fourier")

    def to_real(self, real: bool = True) -> Field:
        """Inverse transform to real space. With real=True the imaginary part is dropped."""
        if self.domain == "real":
            return self
        values = np.fft.ifftn(self.values, axes=self.spatial_axes, norm=FFT_NORM)
        if real:
            values = np.real(values)
        return self.replace(values=values, domain="real")
