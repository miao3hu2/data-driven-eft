from dataclasses import dataclass
import itertools
from core.coarse_graining.base import CoarseGrainer
from core.data.base import Field
import numpy as np


def _kept_slices(n: int, m: int, m_out: int | None = None) -> list[tuple[slice, slice]]:
    """Modes with |k| < m/2 out of n, as (slice in the length-n spectrum, slice in a length-m_out spectrum)
    pairs: the non-negative modes 0, ..., n_pos - 1, then the negative modes -n_neg, ..., -1. m_out defaults to m."""
    m_out = m if m_out is None else m_out
    n_pos = (m + 1) // 2
    n_neg = (m - 1) // 2
    return [(slice(0, n_pos), slice(0, n_pos)), (slice(n - n_neg, n), slice(m_out - n_neg, m_out))]


@dataclass
class BlockCutoffFilter(CoarseGrainer):
    """
    Sharp spectral cutoff keeping |k| < k_max = pi / (block_size * dx) along every spatial axis.

    The cutoff never keeps the coarse Nyquist mode. With odd_grid=True the grid has exactly
    one point per kept mode (m - 1 points for even m)
    """
    block_size: int = 2
    odd_grid: bool = False

    def transform(self, field: Field) -> Field:
        """Keep only the low modes and put them on a grid block_size times coarser."""
        return self._apply(field, reduce_dim=True)

    def project(self, field: Field) -> Field:
        """Zero the high modes but keep the original grid size."""
        return self._apply(field, reduce_dim=False)

    def basis(self, field: Field) -> Field:
        """Basis of the kept modes as a batched Field in the domain of field: values (n_kept, *spatial_shape)."""
        kept = self._kept(field)
        n_space = len(kept)
        basis = np.ones((), dtype=complex)
        for d, ((_, n, _, pairs), dx, k) in enumerate(zip(kept, field.scale, field.wavenumbers())):
            idx = np.concatenate([np.arange(n)[s] for s, _ in pairs])
            if field.domain == "fourier":
                wave = np.eye(n, dtype=complex)[:, idx]
            else:
                wave = np.exp(1j * np.outer(np.arange(n) * dx, k[idx]))  # (n, n_kept on spatial axis d)
            shape = [1] * (2 * n_space) 
            shape[n_space + d], shape[d] = wave.shape # put n_kept_d on axis d and n_d on axis n_space + d
            basis = basis * wave.T.reshape(shape) # shape = (1, ..., n_kept_d, ..., 1, n_d, ..., 1) 
        values = basis.reshape(-1, *field.spatial_shape)
        return Field(values=values, scale=field.scale, batched=True, domain=field.domain)

    def _kept(self, field: Field) -> list[tuple[int, int, int, list[tuple[slice, slice]]]]:
        """Per spatial axis: (axis, n, m, slice pairs) with m the coarse grid size: n // block_size, or the number of kept modes with odd_grid=True."""
        kept = []
        for axis, n in zip(field.spatial_axes, field.spatial_shape):
            m = n // self.block_size
            if m < 1:
                raise ValueError(f"block_size {self.block_size} is larger than the spatial axis of size {n}.")
            m_out = (m + 1) // 2 + (m - 1) // 2 if self.odd_grid else m
            kept.append((axis, n, m_out, _kept_slices(n, m, m_out)))
        return kept

    def _apply(self, field: Field, reduce_dim: bool) -> Field:
        fourier = field.to_fourier()
        values = fourier.values
        ndim = values.ndim

        kept = self._kept(fourier)

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
