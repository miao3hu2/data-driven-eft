from core.inference.base import InferenceMethod
from core.data.base import Field
from core.operators.base import OperatorBasis
from core.rgflow.couplings import CouplingsForOperators


class MaximumLikelihood(InferenceMethod):

    def fit(self, field: Field, basis: OperatorBasis) -> CouplingsForOperators:
        raise NotImplementedError