from abc import ABC, abstractmethod
from core.data.base import Field
import numpy as np


class Operator(ABC):
    name: str

    @abstractmethod
    def evaluate(self, field: Field) -> Field:
        pass


class Identity(Operator):
    name = "I"

    def evaluate(self, field: Field) -> Field:
        return field


class OperatorBasis:

    def __init__(self, operators: list[Operator] | None = None):
        self.operators = list(operators) if operators is not None else []

    def evaluate(self, field: Field) -> list[Field]:
        return [op.evaluate(field) for op in self.operators]

    def stack(self, field: Field) -> Field:
        """O_i(field) for every operator, stacked as a leading component axis: (*lead, *space, n_operators, *components)."""
        fields = self.evaluate(field)
        for op, f in zip(self.operators, fields):
            if f.shape != field.shape:
                raise ValueError(f"operator {op.name} must keep the layout {field.shape} of the field, got {f.shape}.")
            if f.domain != field.domain:
                raise ValueError(f"operator {op.name} must keep the {field.domain} domain of the field, got {f.domain}.")
        values = np.stack([f.values for f in fields], axis=field.n_lead + field.n_space)
        return field.replace(values=values, n_components=field.n_components + 1)

    def add_operator(self, operator: Operator):
        self.operators.append(operator)

    @property
    def names(self):
        return [op.name for op in self.operators]

    def __len__(self):
        return len(self.operators)