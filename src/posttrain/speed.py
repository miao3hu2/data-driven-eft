"""
How fast a basis makes the closure: one step of the Mori-Zwanzig model at the held-out level, modelled and measured.

A step of the closure (KernelEnvironment.rollout_error) takes u from its modes to the grid, evaluates every operator
of the basis on it and takes its modes, then applies the u row of the kernels to the last n_lags states. closure_cost
counts the floating-point operations of that step, so that the reward does not depend on the machine or on what else
it is running:

    FFT of size N           5 N log2 N
    operator O              the modes of u, one inverse FFT per derivative order on the padded grid (2m points),
                            its monomials multiplied there, one FFT back, then the modes of O on the grid of m points
    kernels                 8 n_lags K M (complex multiply-adds of the u row)

measure_speedup times the closure and a direct numerical simulation of the same held-out trajectories on the fine
grid, per snapshot interval (the closure predicts the snapshots after its first n_lags), for verification.
"""

import time

import numpy as np

from core.data.polynomial_pde import PolynomialPDE
from core.operators.symbolic import SymbolicOperator
from core.evaluation.kernels import KernelEnvironment


def _fft(n: int) -> float:
    return 5 * n * np.log2(n)


def operator_cost(op: SymbolicOperator, m: int) -> float:
    """Operations to evaluate one operator on a grid of m points and take its modes."""
    padded = 2 * m
    orders = 1 + max(sum(atom) for atom in op.atoms)  # the derivatives 0..max order, each an inverse FFT
    products = sum(mono.degree for mono in op.monomials) * padded
    return 2 * _fft(m) + (orders + 1) * _fft(padded) + products


def closure_cost(ops: list[SymbolicOperator], m: int, n_lags: int) -> float:
    """Operations of one closure step with observables (u, *ops) on a grid of m points (m modes)."""
    M = 1 + len(ops)
    return _fft(m) + sum(operator_cost(op, m) for op in ops) + 8 * n_lags * m * M


def relative_cost(ops: list[SymbolicOperator], env: KernelEnvironment) -> float:
    """closure_cost relative to the closure with u alone, at the held-out level of env."""
    m, h = env.u_test[env.levels.heldout].shape[-1], env.method.n_lags
    assert h is not None, "the environment fits a fixed number of kernels"
    return closure_cost(ops, m, h) / closure_cost([], m, h)


def measure_speedup(env: KernelEnvironment, ops: list[SymbolicOperator], repeats: int = 3) -> tuple[float | None, float]:
    if not isinstance(env.dynamics, PolynomialPDE):
        raise TypeError("measure_speedup needs a PolynomialPDE to simulate")
    pde, level = env.dynamics, env.levels.heldout
    ops = env.observable_ops(ops)
    u0 = pde.random_initial_states(np.random.default_rng(env.seed + 1), env.n_test)  # those of env.u_test
    simulation = min(_timed(lambda: pde.solve(u0)) for _ in range(repeats)) / (pde.n_time - 1)
    if env.evaluate(ops).heldout_error >= env.max_error:
        return None, simulation
    kernels = env.extrapolated_kernels(ops)
    closure = min(_timed(lambda: env.rollout_error(kernels, level)) for _ in range(repeats))
    return closure / (pde.n_time - kernels.n_lags), simulation


def _timed(f) -> float:
    start = time.perf_counter()
    f()
    return time.perf_counter() - start
