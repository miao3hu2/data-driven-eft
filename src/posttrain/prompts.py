from typing import Callable, Literal

import numpy as np

from core.operators.symbolic import distinct_monomials
from posttrain.reward import MAX_COST, MAX_OPS
from core.evaluation.kernels import KernelEnvironment

Hints = Literal["stats", "list", "data"]

SYSTEM = """You propose operator bases for the effective dynamics of a coarse-grained field.

Data: trajectories of a real scalar field u(x, t) on a periodic domain, from an unknown deterministic evolution law
with random initial states, observed at equally spaced times and coarse-grained at several scales.

Model. A basis {O_1, ..., O_p} defines the observables g = (u, O_1(u), ..., O_p(u)); u is always included. At each
scale, every Fourier mode of the observables follows a Mori-Zwanzig equation with memory,
    g_k(t+1) = sum_l Omega^(l)(k) g_k(t-l) + noise,
whose kernels, Omega^(l)(k), are fitted from correlations. The kernels flow with the scale; they are extrapolated to a coarser scale
and the effective dynamics simulates u there, re-evaluating the operators on the simulated u at every step.

A basis is good if its effective dynamics is accurate and fast: accurate when the operators carry the nonlinearities that u
needs to predict its own future, fast when there are few operators and they are cheap (each one costs FFTs, more for
higher derivatives). Operators linear in u are useless: mode by mode they are proportional to u. An operator and its
x-derivatives are the same observable (D(O) is O again). Such repeats are dropped from your basis and cost reward.

Operators are polynomials in u and its x-derivatives, of any degree and any derivative order, with D for d/dx: D(u) is
u_x, D(u,2) is u_xx, D(u,n) the n-th derivative. Products, sums, positive integer powers and numbers are allowed, as in
the syntax "2*u*D(u,2) - D(u)**3". An operator that would make a step of the effective dynamics cost more than MAX_COST times a
step of u alone is too slow to evaluate: it is ignored and costs reward.

Answer with a quick reasoning, since you will be provided with feedback on your answer. Put the basis between <basis> and </basis>, one operator per line, at most MAX_OPS.
An empty basis (<basis></basis>) means u alone. Format:
<basis>
first operator
second operator
</basis>""".replace("MAX_OPS", str(MAX_OPS)).replace("MAX_COST", f"{MAX_COST:g}")


def diagnostics(env: KernelEnvironment, hints: Hints = "stats", top: int = 8) -> str:
    level = env.levels.fit[0]
    u = env.u[level].values  # (batch, time, m), real
    dt = env.dynamics.dt
    increments = np.diff(u, axis=-1)
    lines = [
        (f"Coarse-grained field on {u.shape[-1]} grid points, {u.shape[0]} trajectories of {u.shape[1]} snapshots "
         f"every {dt:g}:"),
        f"- spatial mean of u: {u[:, 0].mean():+.3f} at t=0, {u[:, -1].mean():+.3f} at t={(u.shape[1] - 1) * dt:g}",
        f"- variance of u: {u[:, 0].var():.3f} at t=0, {u[:, -1].var():.3f} at the end",
        (f"- skewness of u at the end: {_skewness(u[:, -1]):+.2f}; of the increments u(x+dx) - u(x): "
         f"{_skewness(increments[:, -1]):+.2f}"),
        f"- flatness of the increments at the end: {_flatness(increments[:, -1]):.2f} (3 for a Gaussian)",
        "- fraction of the next step of u that u alone does not explain, from fine to coarse: "
        + ", ".join(f"{env.one_step_residual([], n):.3f}" for n in env.levels.fit),
        *unexplained_lines(env),
    ]
    lines += HINTS[hints](env, top)
    return "\n".join(lines)


def relevance_lines(env: KernelEnvironment, top: int = 8) -> list[str]:
    """The monomials of the DSL ranked by how much of what u alone leaves they could explain. It enumerates the whole
    space of operators, which only a finite space allows, and a model can copy the answer from it (the list rule)."""
    pool = distinct_monomials()
    relevance, _, _ = env.candidate_features([], pool, env.levels.fit[0])
    ranked = sorted(zip(relevance, pool), key=lambda r: -r[0])[:top]
    return (["- fraction of that unexplained part each monomial could explain as one more observable:"]
            + [f"    {op.text} ({r:.3f})" for r, op in ranked])


def unexplained_lines(env: KernelEnvironment) -> list[str]:
    """Two statistics that tell task families apart without naming an operator (checked on the 24 tasks and two
    ensembles): the drift of the spatial mean of u, 0 when the evolution conserves the mean (a flux, an x-derivative
    D(...)) and about -1 on the KPZ tasks; and the skewness of the step of u that u alone does not explain, about -2 on
    KPZ, positive where u**2 and u**3 mix, small for Burgers and modified Burgers."""
    level = env.levels.fit[0]
    u = env.u[level].values  # (batch, time, m) on the grid
    mean = u.mean(axis=-1)
    drift = np.mean(mean[:, -1] - mean[:, 0]) / ((u.shape[1] - 1) * env.dynamics.dt) / np.sqrt(np.mean(u**2))
    unexplained = np.real(np.fft.ifft(env.noise([], level)[..., 0], axis=-1))  # on the grid: the modes are all of them
    return [f"- rate of change of the spatial mean of u, relative to the rms of u: {drift:+.3f} per unit time (0 when the "
            f"evolution conserves the mean)",
            f"- skewness of the part of the step of u that u alone does not explain: {_skewness(unexplained):+.2f}"]


SAMPLE_TRAJECTORIES = 3  # trajectories of the data sample of hints="data"


def sample_lines(env: KernelEnvironment, top: int = 8) -> list[str]:
    """A few trajectories of the data themselves, at the coarsest scale, 2 decimals: ~2,500 tokens. For a model large
    enough to read numbers; it can see what the statistics miss (fronts forming, asymmetry, growth or decay)."""
    level = env.levels.predict
    u = env.u[level].values[:SAMPLE_TRAJECTORIES]  # (trajectories, time, m)
    lines = [f"- a sample of the data: {len(u)} trajectories at block size {2**level} ({u.shape[-1]} grid points), u at "
             f"every grid point, one line per snapshot:"]
    for b, path in enumerate(u):
        lines.append(f"    trajectory {b + 1}:")
        lines += [f"      t={n * env.dynamics.dt:.2f}: " + " ".join(f"{v:+.2f}" for v in row) for n, row in enumerate(path)]
    return lines


# what the first prompt adds to the statistics, by the name of LoopConfig.hints: "stats" nothing (no operator named,
# no data shown); "list" the ranked operators, for a small model; "data" a sample of the data, for a larger model
HINTS: dict[str, Callable[[KernelEnvironment, int], list[str]]] = {
    "stats": lambda env, top: [],
    "list": relevance_lines,
    "data": sample_lines,
}


def _skewness(x: np.ndarray) -> float:
    x = x - x.mean()
    return float(np.mean(x**3) / np.mean(x**2) ** 1.5)


def _flatness(x: np.ndarray) -> float:
    x = x - x.mean()
    return float(np.mean(x**4) / np.mean(x**2) ** 2)
