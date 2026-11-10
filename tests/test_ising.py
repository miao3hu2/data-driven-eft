import numpy as np
import pytest

from core.data.ising import Ising2D


@pytest.fixture
def sample_data():
    model = Ising2D(L=4, beta=0.44, seed=42)
    return model.sample(n_samples=5, burn_in=100, thinning=10)


def test_sample_shape(sample_data):
    assert sample_data.values.shape == (5, 4, 4)


def test_sample_scale(sample_data):
    assert sample_data.scale == (1.0, 1.0)


def test_sample_values_are_plus_minus_one(sample_data):
    assert np.all(np.isin(sample_data.values, [-1, 1]))


def test_sample_metadata():
    model = Ising2D(L=4, beta=0.37, J=1.5, seed=1)
    data = model.sample(n_samples=2, burn_in=10, thinning=2)
    assert data.metadata == {"system": "2d_ising", "beta": 0.37, "J": 1.5}


def test_sampling_is_deterministic_given_seed():
    data1 = Ising2D(L=4, beta=0.44, seed=123).sample(n_samples=3, burn_in=50, thinning=5)
    data2 = Ising2D(L=4, beta=0.44, seed=123).sample(n_samples=3, burn_in=50, thinning=5)
    np.testing.assert_array_equal(data1.values, data2.values)


def test_energy_change_matches_manual_calculation():
    spins = np.array(
        [
            [1, 1, -1, 1],
            [-1, 1, 1, -1],
            [1, -1, 1, 1],
            [1, 1, -1, -1],
        ]
    )
    model = Ising2D(L=4, beta=0.44, J=2.0, seed=0)

    # site (0, 0): s=1, periodic neighbors are spins[1,0]=-1, spins[3,0]=1,
    # spins[0,1]=1, spins[0,3]=1 -> nn sum = 2 -> dE = 2*J*s*nn = 2*2*1*2 = 8
    assert model.energy_change(spins, 0, 0) == 8.0

    # site (1, 0): s=-1, periodic neighbors are spins[2,0]=1, spins[0,0]=1,
    # spins[1,1]=1, spins[1,3]=-1 -> nn sum = 2 -> dE = 2*2*(-1)*2 = -8
    assert model.energy_change(spins, 1, 0) == -8.0


def _bond_energy(field):
    from core.operators.ising import NearestNeighbor

    return NearestNeighbor().evaluate(field).values.mean()


def test_swendsen_wang_samples_the_same_distribution_as_metropolis():
    clusters = Ising2D(L=8, beta=0.4, seed=0).sample_clusters(n_samples=400, sweeps=30)
    metropolis = Ising2D(L=8, beta=0.4, seed=1).sample(n_samples=400, burn_in=200, thinning=5)
    assert clusters.values.shape == (400, 8, 8) and np.all(np.isin(clusters.values, [-1, 1]))
    assert _bond_energy(clusters) == pytest.approx(_bond_energy(metropolis), abs=0.04)


def test_swendsen_wang_orders_at_low_temperature_and_not_at_high():
    cold = Ising2D(L=16, beta=0.7, seed=0).sample_clusters(n_samples=50, sweeps=20).values
    hot = Ising2D(L=16, beta=0.2, seed=0).sample_clusters(n_samples=50, sweeps=20).values
    assert np.abs(cold.mean(axis=(1, 2))).mean() > 0.95
    assert np.abs(hot.mean(axis=(1, 2))).mean() < 0.2


def test_pseudo_likelihood_recovers_the_coupling_and_agrees_with_infer_k1():
    from core.inference.ising import InferK1, PseudoLikelihood
    from core.operators.base import OperatorBasis
    from core.operators.ising import NearestNeighbor, NextNearestNeighbor

    samples = Ising2D(L=16, beta=0.35, seed=0).sample_clusters(n_samples=200, sweeps=30)
    k1 = PseudoLikelihood().fit(samples, OperatorBasis([NearestNeighbor()]))["K1"]
    assert k1 == pytest.approx(0.35, abs=0.02)
    assert k1 == pytest.approx(InferK1().fit(samples, OperatorBasis([NearestNeighbor()]))["K1"], abs=1e-3)
    both = PseudoLikelihood().fit(samples, OperatorBasis([NearestNeighbor(), NextNearestNeighbor()]))
    assert both["K1"] == pytest.approx(0.35, abs=0.03) and abs(both["K2"]) < 0.03  # the true Hamiltonian has no K2
