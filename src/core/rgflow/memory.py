from dataclasses import dataclass
import numpy as np
from core.coarse_graining.filter import BlockCutoffFilter
from core.data.base import Field
from core.inference.mori_kernels import MemoryKernels, Origins, mori_kernels_level
from core.inference.mori_zwanzig import MoriZwanzigData, check_trajectories, mori_zwanzig_level
from core.operators.base import OperatorBasis


@dataclass
class RatioData:
    """Training data for gamma_n(t) in Q L u_n[t] ~ gamma_n(t) P L Q L u_n[t]: one row per (level, time step)."""

    time: np.ndarray  # (n_rows,) time coordinate t
    level: np.ndarray  # (n_rows,) filter level n
    block_size: np.ndarray  # (n_rows,) total coarse-graining factor block_size**n
    k_max: np.ndarray  # (n_rows, n_space) cutoff wavenumber along each spatial axis
    ratio: np.ndarray  # (n_rows,) least-squares estimate over all batch members
    stderr: np.ndarray  # (n_rows,) its standard error from the spread over the batch members
    bootstrap: np.ndarray  # (n_rows, n_bootstrap) the estimate on batch members resampled with replacement

    def save(self, path) -> None:
        """Save every column to an .npz file; load it back with RatioData(**np.load(path))."""
        _save_columns(self, path)


@dataclass
class NoiseData:
    """Variance of the noise residual - gamma(t) memory per Fourier mode: one row per (level, time step, wavevector, component). With the forward FFT norm, the variances of one (level, time step) sum over wavevectors to the mean |noise|^2 per grid point."""

    time: np.ndarray  # (n_rows,) time coordinate t
    level: np.ndarray  # (n_rows,) filter level n
    block_size: np.ndarray  # (n_rows,) total coarse-graining factor block_size**n
    k: np.ndarray  # (n_rows, n_space) wavevector of the mode
    component: np.ndarray  # (n_rows,) flat index into the component axes
    noise_var: np.ndarray  # (n_rows,) mean |noise_k|^2 over the batch members

    def save(self, path) -> None:
        """Save every column to an .npz file; load it back with NoiseData(**np.load(path))."""
        _save_columns(self, path)


def ratio_data(data: list[MoriZwanzigData], n_bootstrap: int = 0, seed: int | None = None) -> RatioData:
    """gamma_n(t) for every level and time step of data, from mori_zwanzig_data.

    The bootstrap resamples the batch members of each level and keeps P as fitted on all of them; it shows the spread of the estimate over ensembles of the same size."""
    rng = np.random.default_rng(seed)
    columns = {name: [] for name in RatioData.__dataclass_fields__}
    for d in data:
        gamma, stderr = d.ratio()
        num, den = d.ratio_sums()
        n_batch, n_time = num.shape
        weights = rng.multinomial(n_batch, np.full(n_batch, 1 / n_batch), size=n_bootstrap)  # (n_bootstrap, batch)
        with np.errstate(invalid="ignore", divide="ignore"):
            bootstrap = (weights @ num) / (weights @ den)  # (n_bootstrap, n_time)
        assert d.residual.time is not None
        columns["time"].append(d.residual.time)
        columns["level"].append(np.full(n_time, d.level))
        columns["block_size"].append(np.full(n_time, d.block_size))
        columns["k_max"].append(np.tile(d.k_max, (n_time, 1)))
        columns["ratio"].append(gamma)
        columns["stderr"].append(stderr)
        columns["bootstrap"].append(bootstrap.T)
    return RatioData(**{name: np.concatenate(values) for name, values in columns.items()})


def noise_data(data: list[MoriZwanzigData]) -> NoiseData:
    """Noise variance per Fourier mode for every level and time step of data, from mori_zwanzig_data."""
    columns = {name: [] for name in NoiseData.__dataclass_fields__}
    for d in data:
        noise = d.residual
        assert noise.time is not None
        n_time, n_space = len(noise.time), noise.n_space
        n_components = int(np.prod(noise.component_shape))
        var = d.noise_var().reshape(n_time, -1, n_components)  # (time, wavevector, component)
        k = np.stack(np.meshgrid(*noise.wavenumbers(), indexing="ij"), axis=-1).reshape(-1, n_space)
        shape = var.shape
        columns["time"].append(np.broadcast_to(noise.time[:, None, None], shape).ravel())
        columns["level"].append(np.full(var.size, d.level))
        columns["block_size"].append(np.full(var.size, d.block_size))
        columns["k"].append(np.broadcast_to(k[None, :, None, :], (*shape, n_space)).reshape(-1, n_space))
        columns["component"].append(np.broadcast_to(np.arange(n_components), shape).ravel())
        columns["noise_var"].append(var.ravel())
    return NoiseData(**{name: np.concatenate(values) for name, values in columns.items()})


def _save_columns(columns, path) -> None:
    np.savez(path, **{name: getattr(columns, name) for name in columns.__dataclass_fields__})


def mori_zwanzig_data(
    field: Field,
    n_levels: int,
    block_size: int = 2,
    homogeneous: bool = True,
    odd_grid: bool = True,
    basis: OperatorBasis | None = None,
) -> list[MoriZwanzigData]:
    """Compute Mori-Zwanzig data for every level of coarse-graining of field."""
    check_trajectories(field)

    fourier = field.to_fourier()
    data = []
    for level in range(1, n_levels + 1):
        coarse = BlockCutoffFilter(block_size=block_size**level, odd_grid=odd_grid).transform(fourier)
        if field.domain == "real":
            coarse = coarse.to_real(real=not np.iscomplexobj(field.values))
        k_max = tuple(np.pi / (block_size**level * dx) for dx in field.scale)
        data.append(mori_zwanzig_level(coarse, basis, homogeneous, level=level, block_size=block_size**level, k_max=k_max))
    return data


def mori_kernel_data(
    field: Field,
    n_levels: int,
    block_size: int = 2,
    homogeneous: bool = True,
    odd_grid: bool = True,
    basis: OperatorBasis | None = None,
    n_lags: int | None = None,
    origins: Origins = "initial",
    rcond: float = 1e-10,
) -> list[MemoryKernels]:
    """Markov and memory kernels from two-time correlations for every level of coarse-graining of field."""
    fourier = field.to_fourier()
    data = []
    for level in range(1, n_levels + 1):
        coarse = BlockCutoffFilter(block_size=block_size**level, odd_grid=odd_grid).transform(fourier)
        if field.domain == "real":
            coarse = coarse.to_real(real=not np.iscomplexobj(field.values))
        k_max = tuple(np.pi / (block_size**level * dx) for dx in field.scale)
        data.append(
            mori_kernels_level(
                coarse, basis, homogeneous, n_lags, origins, rcond,
                level=level, block_size=block_size**level, k_max=k_max,
            )
        )
    return data
