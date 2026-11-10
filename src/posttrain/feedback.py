"""
For a given basis, get a text report that answers three questions for the model: which of its operators matter at coarse scales, which don't, and what is still missing.
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from core.operators.symbolic import SymbolicOperator, distinct_monomials
from posttrain.reward import RewardConfig, score_basis
from core.evaluation.kernels import KernelEnvironment

IMPORTANT = 1.02  # an operator is important at the prediction scale if the error without it is at least 2% larger
HARMFUL = 0.98  # and harmful if the error without it is at least 2% smaller


@dataclass
class OperatorFlow:
    text: str  # as written in the basis
    name: str  # canonical name
    share: np.ndarray  # (n_fit + 1,) share of the step of u carried by the operator, fit levels then the prediction scale
    without: float  # error without the operator / with the whole basis

    @property
    def important(self) -> bool:
        return self.without >= IMPORTANT

    @property
    def harmful(self) -> bool:
        return self.without <= HARMFUL

    @property
    def trend(self) -> float:
        """log2 of the share at the prediction scale over the share at the finest fit level. > 0 implies a growing trend towards coarse."""
        return float(np.log2((self.share[-1] + 1e-12) / (self.share[0] + 1e-12)))


@dataclass
class FlowReport:
    texts: tuple[str, ...]  # the operators besides u, as written
    names: tuple[str, ...]
    error: float
    direct_error: float
    relative_cost: float
    reward: float  # accuracy + speed of the reward, without its format part
    block_sizes: tuple[int, ...]  # of the fit levels, then of the prediction scale
    residual: np.ndarray  # (n_fit,) the fraction of the step of u the basis leaves unexplained at each fit level
    operators: list[OperatorFlow] = field(default_factory=list)
    candidates: list[tuple[str, float]] = field(default_factory=list)  # (monomial, relevance) at the coarsest fit level
    extra: list[str] = field(default_factory=list)  # lines from the REPORTS hook, after the rest

    def text(self) -> str:
        sizes = self.block_sizes
        lines = [
            (f"Basis [{', '.join(('u', *self.texts))}]: error {self.error:.4f} at block size {sizes[-1]} (the prediction scale) with the "
             f"extrapolated kernels ({self.direct_error:.4f} with kernels fitted there), cost "
             f"x{self.relative_cost:.2f} of u alone, reward {self.reward:+.3f}."),
            "  unexplained step of u: " + ", ".join(f"{r:.3f} (block size {s})" for r, s in zip(self.residual, sizes)),
        ]
        if self.operators:
            scales = " -> ".join(f"block size {s}" for s in sizes)
            lines.append(f"  RG flow of the share of the step of u each operator carries ({scales}, the last one "
                         f"extrapolated):")
            for op in self.operators:
                trend = "grows" if op.trend > 0.5 else "shrinks" if op.trend < -0.5 else "about constant"
                lines.append(f"    {op.text}: " + " -> ".join(f"{s:.3f}" for s in op.share)
                             + f" ({trend}); without it the error is x{op.without:.2f}")
            important = [op for op in self.operators if op.important]
            harmful = [op for op in self.operators if op.harmful]
            negligible = [op for op in self.operators if not (op.important or op.harmful)]
            if important:
                lines.append(f"  important at block size {sizes[-1]}: " + ", ".join(op.text for op in important))
            if harmful:
                lines.append(f"  harmful at block size {sizes[-1]} (the error is lower without it): "
                             + ", ".join(op.text for op in harmful))
            if negligible:
                lines.append(f"  negligible at block size {sizes[-1]} (dropping one changes the error by less than "
                             f"{IMPORTANT - 1:.0%} and makes the effective dynamics cheaper): "
                             + ", ".join(op.text for op in negligible))
        if self.candidates:
            lines.append(f"  monomials outside the basis that best explain what it leaves at block size {sizes[-2]}, with their "
                         f"relevance:")
            lines.extend(f"    {text} ({r:.3f})" for text, r in self.candidates)
        lines.extend(self.extra)
        return "\n".join(lines)


def operator_shares(env: KernelEnvironment, ops: list[SymbolicOperator], kernels: np.ndarray, level: int) -> np.ndarray:
    """The share of the step of u carried by each operator of ops (the observables after u) with the given kernels
    (n_lags, K, M, M) at a level, on the fit trajectories: (len(ops),)."""
    g = env.basis_modes(ops, level)  # (batch, time, K, M)
    h = kernels.shape[0]
    windows = sliding_window_view(g[:, :-1], h, axis=1)  # (batch, time - h, K, M, h): g(s + w), w = 0 .. h-1
    # Omega^(l) acts on g(t - l): kernels reversed in lag line up with the window, oldest state first
    terms = np.einsum("lkj,bskjl->bskj", kernels[::-1, :, 0, :], windows, optimize=True)  # (batch, time - h, K, M)
    step = np.linalg.norm(g[:, h:, :, 0] - g[:, h - 1:-1, :, 0])
    return np.sqrt(np.sum(np.abs(terms[..., 1:]) ** 2, axis=(0, 1, 2))) / (step + 1e-30)


def flow_report(env: KernelEnvironment, ops: list[SymbolicOperator], config: RewardConfig = RewardConfig(),
                candidates: int = 5, reports: tuple[str, ...] = ()) -> FlowReport:
    """reports: names of REPORTS whose lines end the report, none by default."""
    ops = env.prepare(ops)
    evaluation = env.evaluate(ops)
    reward = score_basis(env, ops, config)
    levels = env.levels
    shares = [operator_shares(env, ops, env.kernels(ops, n).kernels, n) for n in levels.fit]
    shares.append(operator_shares(env, ops, evaluation.extrapolated, levels.predict))
    shares = np.stack(shares, axis=1) if ops else np.zeros((0, len(levels.fit) + 1))  # (p, n_fit + 1)
    operators = [
        OperatorFlow(op.text, op.name, shares[i],
                     env.evaluate([o for o in ops if o is not op]).error / evaluation.error)
        for i, op in enumerate(ops)
    ]
    ranked = ranked_candidates(env, ops, candidates) if candidates else []
    return FlowReport(
        texts=tuple(op.text for op in ops),
        names=tuple(op.name for op in ops),
        error=evaluation.error,
        direct_error=evaluation.direct_error,
        relative_cost=reward.relative_cost,
        reward=reward.accuracy + reward.speed,
        block_sizes=tuple(2**n for n in (*levels.fit, levels.predict)),
        residual=evaluation.residual,
        operators=operators,
        candidates=ranked,
        extra=[line for name in reports for line in REPORTS[name](env, ops)],
    )


def ranked_candidates(env: KernelEnvironment, ops: list[SymbolicOperator], n: int) -> list[tuple[str, float]]:
    """The n monomials outside the basis that could explain most of what it leaves at the coarsest fit level, with
    their relevance. Like prompts.relevance_lines it enumerates the space of operators: replaced in an infinite one."""
    pool = distinct_monomials(ops)
    relevance, _, _ = env.candidate_features(ops, pool, env.levels.fit[-1])
    return [(op.text, float(r)) for r, op in sorted(zip(relevance, pool), key=lambda x: -x[0])[:n]]


def leak_lines(env: KernelEnvironment, ops: list[SymbolicOperator]) -> list[str]:
    """How far the span of the basis is from invariant, observable by observable (KernelEnvironment.leak)."""
    leak = env.leak(ops)
    rows = ", ".join(f"{text} {r:.3f}" for text, r in zip(("u", *(op.text for op in env.prepare(ops))), leak.rows))
    return [f"  share of each observable's step the basis misses at block size {2**leak.level} (0: closed, 1: no better "
            f"than no change): {rows}"]


# hook: extra lines of the feedback on a basis, by the names in LoopConfig.reports. A future entry describes what
# still leaks without naming operators.
REPORTS: dict[str, Callable[[KernelEnvironment, list[SymbolicOperator]], list[str]]] = {"leak": leak_lines}
