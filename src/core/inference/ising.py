from core.inference.base import InferenceMethod
from core.data.base import Field
from core.operators.base import OperatorBasis
from core.rgflow.couplings import CouplingsForOperators
import numpy as np
from scipy.optimize import minimize_scalar


class InferK1(InferenceMethod):

    def fit(self, field: Field, basis: OperatorBasis, beta = 1.0): 
        if basis.names != ["K1"]:
            raise ValueError("this method can only be used to infer the coupling for the operator named K1")
        
        h = self._neighbor_field(field)

        x = field.values

        def negative_log_likelihood(K1):

            z = 2.0 * beta * K1 * h

            loss = np.logaddexp(0.0, - x * z)

            return np.sum(loss)

        result = minimize_scalar(
            negative_log_likelihood,
            bounds=(-2.0, 2.0),
            method="bounded",
        )

        return CouplingsForOperators(basis, result.x, scale=field.scale) # pyright: ignore[reportAttributeAccessIssue]


    def _neighbor_field(self, field: Field) -> np.ndarray:

        x = field.values

        x_neighbor = np.zeros(x.shape, dtype=x.dtype)
        for axis in field.spatial_axes:
            x_neighbor += np.roll(x, 1, axis=axis) + np.roll(x, -1, axis=axis)

        return x_neighbor