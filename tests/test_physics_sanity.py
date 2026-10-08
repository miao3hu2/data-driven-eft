"""Physics sanity checks.

Unlike the other test modules (which pin down "does the code match its own
formula"), these tests pin down "does the code respect the physics it claims
to model" -- symmetries of the zero-field 2D Ising Hamiltonian, exact
bounds implied by the lattice, and qualitative thermodynamic behavior across
the phase transition. A regression here means the implementation still
computes *something*, but no longer represents 2D Ising physics.
"""

import numpy as np

from core.coarse_graining.block import MajorityBlockSpin
from core.data.base import Field
from core.data.ising import Ising2D
from core.operators.ising import NearestNeighbor, NextNearestNeighbor
from evaluation.observables import magnetization, susceptibility


def _total(field):
    """Spatial sum of an operator's density field: the lattice bond sum."""
    return field.values.sum(axis=field.spatial_axes)


# Onsager's exact critical point for the 2D square-lattice Ising model
# (J=1, no external field). Used only to pick beta values that are
# unambiguously deep in the ordered/disordered phase, not to check its value.
BETA_C = np.log(1 + np.sqrt(2)) / 2


def _random_spins(shape, seed):
    return np.random.default_rng(seed).choice([-1, 1], size=shape)


# --- Z2 (global spin-flip) symmetry -----------------------------------------
#
# The zero-field Ising Hamiltonian only couples spins in pairs (s_i * s_j),
# so flipping every spin in the lattice leaves the energy, and any operator
# built purely from pairwise products, exactly unchanged.

def test_energy_change_invariant_under_global_spin_flip():
    spins = _random_spins((6, 6), seed=0)
    model = Ising2D(L=6, beta=0.44, J=1.3, seed=0)

    for i in range(6):
        for j in range(6):
            assert model.energy_change(spins, i, j) == model.energy_change(-spins, i, j)


def test_nearest_neighbor_invariant_under_global_spin_flip():
    spins = _random_spins((6, 6), seed=1)
    op = NearestNeighbor()

    assert _total(op.evaluate(Field(values=spins, scale=1.0))) == _total(op.evaluate(Field(values=-spins, scale=1.0)))


def test_next_nearest_neighbor_invariant_under_global_spin_flip():
    spins = _random_spins((6, 6), seed=2)
    op = NextNearestNeighbor()

    assert _total(op.evaluate(Field(values=spins, scale=1.0))) == _total(op.evaluate(Field(values=-spins, scale=1.0)))


# --- Translation invariance (periodic boundary conditions) -----------------
#
# On a periodic lattice, shifting every spin by a fixed offset relabels the
# bonds but doesn't change which pairs are bonded, so any lattice-sum
# operator must be unchanged by an arbitrary periodic roll.

def test_nearest_neighbor_invariant_under_translation():
    spins = _random_spins((6, 6), seed=3)
    shifted = np.roll(spins, shift=(2, 5), axis=(0, 1))
    op = NearestNeighbor()

    assert _total(op.evaluate(Field(values=spins, scale=1.0))) == _total(op.evaluate(Field(values=shifted, scale=1.0)))


def test_next_nearest_neighbor_invariant_under_translation():
    spins = _random_spins((6, 6), seed=4)
    shifted = np.roll(spins, shift=(1, 4), axis=(0, 1))
    op = NextNearestNeighbor()

    assert _total(op.evaluate(Field(values=spins, scale=1.0))) == _total(op.evaluate(Field(values=shifted, scale=1.0)))


# --- Point-group symmetry of the square lattice -----------------------------
#
# A 90-degree rotation maps horizontal bonds onto vertical bonds (and one
# lattice diagonal onto the other), but K1/K2 sum over both directions
# already, so the total must be invariant under rotation too.

def test_nearest_neighbor_invariant_under_90_degree_rotation():
    spins = _random_spins((6, 6), seed=5)
    op = NearestNeighbor()

    assert _total(op.evaluate(Field(values=spins, scale=1.0))) == _total(op.evaluate(Field(values=np.rot90(spins), scale=1.0)))


def test_next_nearest_neighbor_invariant_under_90_degree_rotation():
    spins = _random_spins((6, 6), seed=6)
    op = NextNearestNeighbor()

    assert _total(op.evaluate(Field(values=spins, scale=1.0))) == _total(op.evaluate(Field(values=np.rot90(spins), scale=1.0)))


# --- Exact bounds ------------------------------------------------------------

def test_nearest_neighbor_bond_sum_is_bounded_by_bond_count():
    # A periodic L x L lattice has exactly 2*L*L nearest-neighbor bonds
    # (L*L horizontal + L*L vertical), and each bond contributes s_i*s_j in
    # [-1, 1], so |K1| can never exceed 2*L*L regardless of spin values.
    L = 6
    spins = _random_spins((L, L), seed=7)
    op = NearestNeighbor()

    assert abs(_total(op.evaluate(Field(values=spins, scale=1.0)))) <= 2 * L * L


def test_magnetization_is_bounded():
    # Magnetization is a mean of +-1 valued spins, so it must lie in [-1, 1]
    # for any physically sampled configuration.
    data = Ising2D(L=8, beta=0.44, seed=8).sample(n_samples=10, burn_in=200, thinning=5)
    m = magnetization(data)

    assert np.all(m >= -1.0) and np.all(m <= 1.0)


def test_susceptibility_is_non_negative():
    # susceptibility = beta * N * Var(m); a variance can't be negative, so
    # neither can the susceptibility for any physical (beta > 0) sample set.
    data = Ising2D(L=8, beta=0.44, seed=9).sample(n_samples=10, burn_in=200, thinning=5)

    assert susceptibility(data, beta=0.44) >= 0.0


# --- Coarse-graining / RG consistency ---------------------------------------

def test_coarse_graining_preserves_saturated_ferromagnetic_state():
    # The fully ordered ground states (all spins up, or all spins down) are
    # fixed points of majority-rule block-spin coarse-graining: every block
    # is unanimous, so no tie-breaking is ever invoked.
    cg = MajorityBlockSpin(block_size=2)

    all_up = cg.transform(Field(values=np.ones((8, 8)), scale=1.0))
    assert np.all(all_up.values == 1)

    all_down = cg.transform(Field(values=-np.ones((8, 8)), scale=1.0))
    assert np.all(all_down.values == -1)


def test_coarse_graining_conserves_physical_system_size():
    # Block-spin RG trades resolution for coarseness -- it must not change
    # the physical extent of the system (lattice size * lattice spacing).
    L = 8
    field = Field(values=_random_spins((L, L), seed=10), scale=1.0)
    blocked = MajorityBlockSpin(block_size=2).transform(field)

    original_extent = L * field.scale[0]
    new_L = blocked.values.shape[0]
    new_extent = new_L * blocked.scale[0]

    assert new_extent == original_extent


# --- Thermodynamics across the phase transition -----------------------------

def test_low_temperature_is_more_ordered_than_high_temperature():
    # Onsager's exact result puts the ferromagnetic transition at BETA_C.
    # Deep in the ordered phase (beta >> BETA_C) the system should be far
    # more magnetized on average than deep in the disordered phase
    # (beta << BETA_C). Checked as a relative comparison (rather than
    # against fixed thresholds) to keep the test robust to finite-size
    # effects while still being a meaningful physics regression check.
    assert 0.05 < BETA_C < 1.0

    ordered = Ising2D(L=8, beta=1.0, seed=11).sample(n_samples=20, burn_in=500, thinning=5)
    disordered = Ising2D(L=8, beta=0.05, seed=11).sample(n_samples=20, burn_in=500, thinning=5)

    mean_abs_m_ordered = np.mean(np.abs(magnetization(ordered)))
    mean_abs_m_disordered = np.mean(np.abs(magnetization(disordered)))

    assert mean_abs_m_ordered > mean_abs_m_disordered
