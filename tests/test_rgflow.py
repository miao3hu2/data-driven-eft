import numpy as np
import pytest

from core.data.base import Field
from core.effective_theory.base import EffectiveTheory
from core.inference.mori_zwanzig import mori_zwanzig_level
from core.operators.base import Operator, OperatorBasis
from core.rgflow.flow import RGFlow
from core.rgflow.couplings import CouplingsForOperators


class _StubOperator(Operator):
    def __init__(self, name):
        self.name = name

    def evaluate(self, field):
        return field


def _couplings(K, scale):
    basis = OperatorBasis([_StubOperator("K")])
    return CouplingsForOperators(basis, np.array([K]), scale=scale)


def _theory(K, scale):
    return EffectiveTheory(OperatorBasis([_StubOperator("K")]), _couplings(K, scale))


def _dynamics(seed):
    rng = np.random.default_rng(seed)
    field = Field(values=rng.normal(size=(200, 5, 3, 3)), scale=1.0, batched=True, time=np.arange(5))
    return mori_zwanzig_level(field).dynamics()


def test_flow_len_and_scales():
    flow = RGFlow(points=[_theory(0.44, 1), _theory(0.42, 2)])
    assert len(flow) == 2
    assert flow.scales == [1, 2]


def test_flow_values_returns_per_point_coupling():
    flow = RGFlow(points=[_theory(0.44, 1), _theory(0.42, 2)])
    assert flow.values("K") == [0.44, 0.42]


def test_add_appends_a_point():
    flow = RGFlow(points=[_theory(0.44, 1)])
    flow.add(_theory(0.42, 2))
    assert len(flow) == 2
    assert flow.scales == [1, 2]


def test_empty_flow_has_no_points():
    for flow in (RGFlow(points=[]), RGFlow()):
        assert len(flow) == 0
        assert flow.scales == []


def test_values_raises_keyerror_for_missing_coupling():
    flow = RGFlow(points=[_theory(0.44, 1)])
    with pytest.raises(KeyError):
        flow.values("h")


def test_points_evaluate_as_theories():
    flow = RGFlow(points=[_theory(0.44, 1), _theory(0.42, 2)])
    field = Field(values=np.ones((2, 2)), scale=1.0)
    np.testing.assert_allclose(flow.points[1].evaluate(field).values, 0.42)


def test_dynamics_flow_gives_every_part():
    points = [_dynamics(0), _dynamics(1)]
    flow = RGFlow(points=points)
    for part, name, attribute in [
        ("couplings", "I", "markov"),
        ("memory", "I", "memory"),
        ("ratio", "gamma", "ratio"),
        ("noise", "noise_var", "noise"),
    ]:
        for value, p in zip(flow.values(name, part=part), points):
            np.testing.assert_array_equal(value, getattr(p, attribute)[name])


def test_dynamics_parts_need_dynamics_points():
    flow = RGFlow(points=[_dynamics(0), _theory(0.44, 2)])
    with pytest.raises(ValueError, match="EffectiveDynamics"):
        flow.values("gamma", part="ratio")
    with pytest.raises(ValueError, match="part must be"):
        flow.values("K", part="markov")


def test_bare_couplings_are_wrapped_with_a_deprecation_warning():
    with pytest.warns(DeprecationWarning, match="EffectiveTheory"):
        flow = RGFlow([_couplings(0.44, 1)])
    with pytest.warns(DeprecationWarning):
        flow.add(_couplings(0.42, 2))
    assert all(isinstance(p, EffectiveTheory) for p in flow.points)
    assert flow.values("K") == [0.44, 0.42]
    assert flow.scales == [1, 2]
    assert flow.points[0].basis.names == ["K"]
    # the operators are known only by name, so the theory cannot be evaluated
    with pytest.raises(NotImplementedError, match="only known by name"):
        flow.points[0].evaluate(Field(values=np.ones((2, 2)), scale=1.0))


def test_other_points_are_rejected():
    with pytest.raises(TypeError, match="EffectiveTheory"):
        RGFlow(points=[0.44])
