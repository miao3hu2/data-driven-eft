from dataclasses import dataclass
import itertools
from core.coarse_graining.base import CoarseGrainer
from core.data.base import Field
import numpy as np


def _kept_slices(n: int, m: int) -> list[tuple[slice, slice]]:
    """Modes with |k| < m/2 out of n, as (slice in the length-n spectrum, slice in a length-m spectrum)
    pairs: the non-negative modes 0, ..., n_pos - 1, then the negative modes -n_neg, ..., -1.
    """
    n_pos = (m + 1) // 2
    n_neg = (m - 1) // 2
    return [(slice(0, n_pos), slice(0, n_pos)), (slice(n - n_neg, n), slice(m - n_neg, m))]


@dataclass
class BlockCutoffFilter(CoarseGrainer):
    """Sharp spectral cutoff keeping |k| < k_max = pi / (block_size * dx) along every spatial axis.
    """
    block_size: int = 2

    def transform(self, field: Field) -> Field:
        """Keep only the low modes and put them on a grid block_size times coarser."""
        return self._apply(field, reduce_dim=True)

    def project(self, field: Field) -> Field:
        """Zero the high modes but keep the original grid."""
        return self._apply(field, reduce_dim=False)

    def _apply(self, field: Field, reduce_dim: bool) -> Field:
        fourier = field.to_fourier()
        values = fourier.values
        ndim = values.ndim

        kept = []  # per spatial axis: (axis, n, m, slice pairs)
        for axis, n in zip(fourier.spatial_axes, fourier.spatial_shape):
            m = n // self.block_size
            if m < 1:
                raise ValueError(f"block_size {self.block_size} is larger than the spatial axis of size {n}.")
            kept.append((axis, n, m, _kept_slices(n, m)))

        if reduce_dim:
            coarse_shape = list(values.shape)
            for axis, _, m, _ in kept:
                coarse_shape[axis] = m
            coarse = np.zeros(coarse_shape, dtype=values.dtype)
            for corner in itertools.product(*(pairs for *_, pairs in kept)):
                src = [slice(None)] * ndim
                dst = [slice(None)] * ndim
                for (axis, *_), (s, d) in zip(kept, corner):
                    src[axis], dst[axis] = s, d
                coarse[tuple(dst)] = values[tuple(src)]
            values = coarse
            new_scale = tuple(dx * n / m for (_, n, m, _), dx in zip(kept, fourier.scale))
        else:
            # copy first to avoid mutating the caller's data
            if field.domain == "fourier":
                values = values.copy()
            for axis, n, _, ((pos, _), (neg, _)) in kept:
                cut = [slice(None)] * ndim
                cut[axis] = slice(pos.stop, neg.start)
                values[tuple(cut)] = 0
            new_scale = tuple(fourier.scale)

        filtered = fourier.replace(values=values, scale=new_scale)

        if field.domain == "real":
            return filtered.to_real(real=not np.iscomplexobj(field.values))
        return filtered
