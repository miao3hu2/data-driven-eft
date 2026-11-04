from typing import Literal

import numpy as np

from core.operators.symbolic import SymbolicOperator, all_monomials
from posttrain.tasks import DEMO, EnvSize, Task, environment
from core.evaluation.kernels import KernelEnvironment

Hints = Literal["stats", "relevance"]

SYSTEM = """You propose closures for coarse-grained dynamics.

Data: trajectories of a real scalar field u(x, t) on a periodic domain, from an unknown deterministic evolution law
with random initial states, observed at equally spaced times and coarse-grained at several scales.

Model. A basis {O_1, ..., O_p} defines the observables g = (u, O_1(u), ..., O_p(u)); u is always included. At each
scale, every Fourier mode of the observables follows a Mori-Zwanzig equation with memory,
    g_k(t+1) = sum_l Omega^(l)(k) g_k(t-l) + noise,
whose kernels are fitted from correlations. The kernels flow with the scale; they are extrapolated to a coarser scale
and the closure simulates u there, re-evaluating the operators on the simulated u at every step.

A basis is good if the closure is accurate and fast: accurate when the operators carry the nonlinearities that u
needs to predict its own future, fast when there are few operators and they are cheap (each one costs FFTs, more for
higher derivatives). Linear operators (D(u), D(u,2), ...) are useless: mode by mode they are proportional to u. An
operator and its x-derivatives are the same observable (D(u**2) is u**2 again).

Operators are polynomials in u and its x-derivatives, with D for d/dx: D(u) is u_x, D(u,2) is u_xx, up to order 4.
Products, sums, integer powers and numbers are allowed, e.g. "u**2", "D(u)**2", "u*D(u,2) - D(u)**2". Degree at most 3.

Answer with a short reasoning, then the basis between <basis> and </basis>, one operator per line, at most 4.
An empty basis (<basis></basis>) means u alone. Example:
<basis>
u**2
</basis>"""


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
    ]
    if hints == "relevance":
        pool = [SymbolicOperator(text) for text in all_monomials()]
        pool = [op for op in pool if op.degree > 1]
        relevance, _, _ = env.candidate_features([], pool, level)
        ranked = sorted(zip(relevance, pool), key=lambda r: -r[0])[:top]
        lines.append("- fraction of that unexplained part each monomial could explain as one more observable: "
                     + ", ".join(f"{op.text} {r:.3f}" for r, op in ranked))
    return "\n".join(lines)


def task_prompt(task: Task, size: EnvSize = DEMO, hints: Hints = "stats") -> str:
    return diagnostics(environment(task, size), hints) + "\n\nPropose the basis."


def messages(task: Task, size: EnvSize = DEMO, hints: Hints = "stats", system_role: bool = True) -> list[dict]:
    """The chat prompt of a task; system_role=False folds the system prompt into the user turn, for chat templates without a system role."""
    user = task_prompt(task, size, hints)
    if system_role:
        return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    return [{"role": "user", "content": SYSTEM + "\n\n" + user}]


def _skewness(x: np.ndarray) -> float:
    x = x - x.mean()
    return float(np.mean(x**3) / np.mean(x**2) ** 1.5)


def _flatness(x: np.ndarray) -> float:
    x = x - x.mean()
    return float(np.mean(x**4) / np.mean(x**2) ** 2)
