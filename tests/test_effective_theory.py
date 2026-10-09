import numpy as np
import pytest

from core.data.base import Field
from core.effective_theory.base import EffectiveDynamics, EffectiveTheory
from core.operators.base import Identity, OperatorBasis
from core.operators.ising import NearestNeighbor
from core.rgflow.couplings import CouplingsForOperators


def test_effective_theory_accepts_matching_couplings():
    basis = OperatorBasis([NearestNeighbor()])
    theory = EffectiveTheory(basis, CouplingsForOperators(basis, np.array([0.44]), scale=1.0))
    assert theory.describe() == "H_eff = +0.440000 * K1"


def test_effective_theory_rejects_mismatched_couplings():
    basis = OperatorBasis([NearestNeighbor()])
    with pytest.raises(ValueError, match="mismatch"):
        EffectiveTheory(basis, CouplingsForOperators(["K2"], np.array([0.44]), scale=1.0))


def _dynamics(**changes):
    basis = OperatorBasis([Identity()])
    fields = {
        "basis": basis,
        "couplings": CouplingsForOperators(basis, [np.full((3, 1), -0.5)], scale=1.0),
        "memory": CouplingsForOperators(basis, [np.full((3, 1), 0.2)], scale=1.0),
        "ratio": CouplingsForOperators(["gamma"], [np.array([0.0, 1.0, 2.0])], scale=1.0),
        "noise": CouplingsForOperators(["noise_var"], [np.ones((3, 3))], scale=1.0),
        "time": np.arange(3.0),
        "wavevectors": np.array([[0.0], [1.0], [-1.0]]),
        "homogeneous": True,
    }
    return EffectiveDynamics(**{**fields, **changes})


def test_effective_dynamics_drift_combines_markov_and_memory():
    dynamics = _dynamics()
    np.testing.assert_allclose(dynamics.drift(0)["I"], -0.5)
    np.testing.assert_allclose(dynamics.drift(2)["I"], -0.5 + 2 * 0.2)
    assert dynamics.describe() == "du = (-0.500000±0.000000 [3, 1] + gamma(t) +0.200000±0.000000 [3, 1]) * I + noise"


def test_effective_dynamics_checks_its_couplings():
    with pytest.raises(ValueError, match="mismatch"):
        _dynamics(memory=CouplingsForOperators(["v"], [np.zeros((3, 1))], scale=1.0))
    with pytest.raises(ValueError, match="gamma"):
        _dynamics(ratio=CouplingsForOperators(["ratio"], [np.zeros(3)], scale=1.0))


def test_effective_theory_evaluates_the_hamiltonian_density():
    x = np.random.default_rng(0).choice([-1.0, 1.0], size=(4, 5, 5))
    field = Field(values=x, scale=1.0, batched=True)
    basis = OperatorBasis([NearestNeighbor()])
    theory = EffectiveTheory(basis, CouplingsForOperators(basis, np.array([0.44]), scale=1.0))
    np.testing.assert_allclose(theory.evaluate(field).values, 0.44 * NearestNeighbor().evaluate(field).values)


def test_effective_theory_evaluate_needs_one_number_per_operator():
    basis = OperatorBasis([Identity()])
    theory = EffectiveTheory(basis, CouplingsForOperators(basis, [np.ones(3)], scale=1.0))
    with pytest.raises(ValueError, match="one coupling"):
        theory.evaluate(Field(values=np.ones((2, 3)), scale=1.0, batched=True))


def test_effective_dynamics_is_an_effective_theory():
    dynamics = _dynamics()
    assert isinstance(dynamics, EffectiveTheory)
    assert dynamics.markov is dynamics.couplings
