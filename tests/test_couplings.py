import numpy as np
import pytest

from core.operators.base import Operator, OperatorBasis
from core.rgflow.couplings import CouplingsForOperators


class _StubOperator(Operator):
    def __init__(self, name):
        self.name = name

    def evaluate(self, field):
        return field


def _basis(*names):
    return OperatorBasis([_StubOperator(name) for name in names])


def test_getitem_returns_value():
    c = CouplingsForOperators(_basis("K", "h"), np.array([0.44, 0.1]), scale=1.0)
    assert c["K"] == 0.44
    assert c["h"] == 0.1


def test_get_returns_default_when_missing():
    c = CouplingsForOperators(_basis("K"), np.array([0.44]), scale=1.0)
    assert c.get("missing") is None
    assert c.get("missing", 0.0) == 0.0
    assert c.get("K", 0.0) == 0.44


def test_as_array_preserves_insertion_order_and_values():
    c = CouplingsForOperators(_basis("K", "h", "g"), np.array([0.44, 0.1, -1.5]), scale=1.0)
    np.testing.assert_array_equal(c.as_array(), [0.44, 0.1, -1.5])
    assert c.as_array().dtype == np.float64


def test_raises_on_values_basis_length_mismatch():
    with pytest.raises(ValueError):
        CouplingsForOperators(_basis("K", "h"), np.array([0.44]), scale=1.0)


def test_raises_on_duplicate_operator_names():
    with pytest.raises(ValueError):
        CouplingsForOperators(_basis("K", "K"), np.array([0.44, 0.1]), scale=1.0)


def test_couplings_can_be_arrays_of_different_shapes():
    c = CouplingsForOperators(["u", "u2"], [np.arange(3.0), np.ones((2, 2))], scale=1.0)
    np.testing.assert_array_equal(c["u"], [0.0, 1.0, 2.0])
    assert c["u2"].shape == (2, 2)
    assert c.names == ["u", "u2"]


def test_array_values_give_one_coupling_per_leading_entry():
    c = CouplingsForOperators(_basis("K", "h"), np.array([[1.0, 2.0], [3.0, 4.0]]), scale=1.0)
    np.testing.assert_array_equal(c["h"], [3.0, 4.0])
    assert c.as_array().shape == (2, 2)


def test_as_array_keeps_complex_couplings():
    c = CouplingsForOperators(["u"], [np.array([1 + 2j, 3j])], scale=1.0)
    assert np.iscomplexobj(c.as_array())


def test_stderr_must_name_operators_of_the_basis():
    c = CouplingsForOperators(["gamma"], [np.array([0.0, 0.5])], scale=1.0, stderr={"gamma": [0.0, 0.1]})
    assert c.stderr is not None
    np.testing.assert_array_equal(c.stderr["gamma"], [0.0, 0.1])
    with pytest.raises(ValueError):
        CouplingsForOperators(["gamma"], [np.array([0.0])], scale=1.0, stderr={"other": [0.1]})
