import numpy as np

from core.data.base import Field


def magnetization(field: Field) -> np.ndarray:

    spins = field.values

    return spins.mean(axis=field.spatial_axes)



def mean_magnetization(field: Field) -> float:
    return float(
        np.mean(magnetization(field))
    )


def susceptibility(field: Field, beta: float) -> float:

    m = magnetization(field)

    N = np.prod(field.spatial_shape)

    return beta * N * np.var(m)
