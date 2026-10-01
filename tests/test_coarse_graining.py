import numpy as np

from core.data.base import Field
from core.data.ising import Ising2D
from core.coarse_graining.block import MajorityBlockSpin
import pytest

def test_majority_block_spin():

    ising_model = Ising2D(L=8, beta=0.44, seed=42)
    data = ising_model.sample(n_samples=10, burn_in=100, thinning=10)
    cg = MajorityBlockSpin(block_size=2)
    blocked_data1 = cg.transform(data)
    blocked_data2 = cg.transform(blocked_data1)

    assert blocked_data1.values.shape == (10, 4, 4)
    assert blocked_data1.scale == (2.0, 2.0)
    assert blocked_data2.values.shape == (10, 2, 2)
    assert blocked_data2.scale == (4.0, 4.0)


def test_majority_block_spin_non_batched_values():
    # Blocks (block_size=2), by (row, col) index:
    #   (0,0): [[1,1],[1,-1]]  sum= 2 -> majority  1
    #   (0,1): [[1,-1],[-1,-1]] sum=-2 -> majority -1
    #   (1,0): [[1,-1],[-1,1]]  sum= 0 -> tie, broken randomly
    #   (1,1): [[-1,-1],[-1,-1]] sum=-4 -> majority -1
    values = np.array(
        [
            [1, 1, 1, -1],
            [1, -1, -1, -1],
            [1, -1, -1, -1],
            [-1, 1, -1, -1],
        ]
    )
    field = Field(values=values, scale=1.0)
    cg = MajorityBlockSpin(block_size=2)
    blocked = cg.transform(field)

    assert blocked.values[0, 0] == 1
    assert blocked.values[0, 1] == -1
    assert blocked.values[1, 0] in (-1, 1)
    assert blocked.values[1, 1] == -1
    assert blocked.scale == (2.0, 2.0)


def test_majority_block_spin_batched_values():
    base = np.array(
        [
            [1, 1, 1, -1],
            [1, -1, -1, -1],
            [1, -1, -1, -1],
            [-1, 1, -1, -1],
        ]
    )
    values = np.stack([base, -base])
    field = Field(values=values, scale=1.0, batched=True)
    cg = MajorityBlockSpin(block_size=2)
    blocked = cg.transform(field)

    # Non-tie blocks mirror the flip; the tie block (1, 0) is broken randomly
    np.testing.assert_array_equal(blocked.values[0, [0, 0, 1], [0, 1, 1]], [1, -1, -1])
    np.testing.assert_array_equal(blocked.values[1, [0, 0, 1], [0, 1, 1]], [-1, 1, 1])
    assert set(blocked.values[:, 1, 0]) <= {-1, 1}
    assert blocked.scale == (2.0, 2.0)

def test_majority_block_spin_with_time_axis():
    base = np.array(
        [
            [1, 1, 1, -1],
            [1, -1, -1, -1],
            [1, -1, -1, -1],
            [-1, 1, -1, -1],
        ]
    )
    values = np.stack([np.stack([base, -base])] * 3)  # (batch=3, time=2, 4, 4)
    field = Field(values=values, scale=1.0, batched=True, time=[0.0, 1.0])
    blocked = MajorityBlockSpin(block_size=2).transform(field)

    assert blocked.values.shape == (3, 2, 2, 2)
    # Non-tie blocks (0,0), (0,1), (1,1) per time step; the tie block (1, 0) is broken randomly
    rows, cols = [0, 0, 1], [0, 1, 1]
    np.testing.assert_array_equal(blocked.values[:, 0, rows, cols], [[1, -1, -1]] * 3)
    np.testing.assert_array_equal(blocked.values[:, 1, rows, cols], [[-1, 1, 1]] * 3)
    assert set(blocked.values[:, :, 1, 0].ravel()) <= {-1, 1}
    np.testing.assert_array_equal(blocked.time, [0.0, 1.0])


@pytest.mark.parametrize("shape, block_size", [((5, 5), 2), ((8, 8), 3), ((4, 6), 4)])
def test_majority_block_spin_rejects_indivisible_shape(shape, block_size):
    field = Field(values=np.ones(shape), scale=1.0)
    with pytest.raises(ValueError, match="not divisible"):
        MajorityBlockSpin(block_size=block_size).transform(field)


def test_majority_block_spin_odd_size_divisible_by_block():
    field = Field(values=np.ones((9, 9)), scale=1.0)
    blocked = MajorityBlockSpin(block_size=3).transform(field)
    assert blocked.values.shape == (3, 3)


def test_majority_block_spin_random_tie_break_is_unbiased():
    # Every 2x2 block is a tie: [[1,-1],[-1,1]] tiled
    values = np.tile([[1, -1], [-1, 1]], (100, 100))
    field = Field(values=values, scale=1.0)
    blocked = MajorityBlockSpin(block_size=2, seed=0).transform(field)

    assert set(np.unique(blocked.values)) == {-1, 1}
    # 10_000 fair coin flips: mean should be well within 5 sigma (0.05) of zero
    assert abs(blocked.values.mean()) < 0.05


def test_majority_block_spin_seed_is_reproducible():
    values = np.tile([[1, -1], [-1, 1]], (8, 8))
    field = Field(values=values, scale=1.0)
    a = MajorityBlockSpin(block_size=2, seed=123).transform(field)
    b = MajorityBlockSpin(block_size=2, seed=123).transform(field)
    np.testing.assert_array_equal(a.values, b.values)
