from core.inference.base import InferenceMethod
from core.data.base import Field
from core.operators.base import OperatorBasis
from core.rgflow.couplings import CouplingsForOperators
import numpy as np
from scipy.optimize import minimize, minimize_scalar


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


class PseudoLikelihood(InferenceMethod):
    """The couplings K_a of H = -sum_a K_a O_a (in units of k_B T) of spin configurations, by maximum pseudo-likelihood.

    Each spin given all the others follows P(s_i | rest) = 1 / (1 + exp(-2 s_i h_i)), with the local field
    h_i = sum_a K_a dO_a/ds_i. The pseudo-likelihood, the product of these over sites and samples, is a logistic
    regression in K: concave, with a unique maximum, and consistent for any number of operators. Every operator of the
    basis needs a local_field method (NearestNeighbor, NextNearestNeighbor). With the single operator K1 it is InferK1
    at beta = 1."""

    def fit(self, field: Field, basis: OperatorBasis) -> CouplingsForOperators:
        missing = [op.name for op in basis.operators if not hasattr(op, "local_field")]
        if missing:
            raise ValueError(f"operators {missing} have no local_field: they are not pair couplings of spins")
        s = field.values
        X = np.stack([2.0 * s * op.local_field(field) for op in basis.operators], axis=-1).reshape(-1, len(basis))

        def loss(K):
            z = X @ K
            # d/dK sum log(1 + exp(-z)) = -X^T sigmoid(-z)
            return np.logaddexp(0.0, -z).sum(), -X.T @ np.exp(-np.logaddexp(0.0, z))

        result = minimize(loss, np.zeros(len(basis)), jac=True, method="L-BFGS-B")
        return CouplingsForOperators(basis, result.x, scale=field.scale)
