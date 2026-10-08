import numpy as np
from core.data.base import Field
from core.operators.base import Operator
from itertools import combinations


class NearestNeighbor(Operator):

    name = "K1"

    def evaluate(self, field: Field) -> Field:

        x = field.values
        spatial_axes = field.spatial_axes

        bonds = sum(x * np.roll(x, -1, axis=axis) for axis in spatial_axes)

        return field.replace(values=np.asarray(bonds))


class NextNearestNeighbor(Operator):

    name = "K2"

    def evaluate(self, field: Field) -> Field:

        x = field.values
        spatial_axes = field.spatial_axes

        if len(spatial_axes) < 2:
            raise ValueError("NextNearestNeighbor operator requires at least 2 spatial dimensions.")

        axes = list(combinations(spatial_axes, 2))

        bonds = sum(
            x * np.roll(x, (-1, -1), axis=axis) + x * np.roll(x, (-1, 1), axis=axis)
            for axis in axes
        )

        return field.replace(values=np.asarray(bonds))
