"""
For a given basis, get a text report that answers three questions for the model: which of its operators matter at coarse scales, which don't, and what is still missing.
"""

from dataclasses import dataclass, field

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from core.operators.symbolic import SymbolicOperator, all_monomials
from posttrain.reward import RewardConfig, score_basis
from core.evaluation.kernels import KernelEnvironment

IMPORTANT = 1.02  # an operator is important at the held-out level if the error without it is at least 2% larger
HARMFUL = 0.98  # and harmful if the error without it is at least 2% smaller


@dataclass
class OperatorFlow:
    text: str  # as written in the basis
    name: str  # canonical name
    share: np.ndarray  # (n_fit + 1,) share of the step of u carried by the operator, fit levels then the held-out level
    without: float  # held-out error without the operator / with the whole basis

    @property
    def important(self) -> bool:
        return self.without >= IMPORTANT

    @property
    def harmful(self) -> bool:
        return self.without <= HARMFUL

    @property
    def trend(self) -> float:
        """log2 of the share at the held-out level over the share at the finest fit level. > 0 implies a growing trend towards coarse."""
        return float(np.log2((self.share[-1] + 1e-12) / (self.share[0] + 1e-12)))


@dataclass
class FlowReport:
    texts: tuple[str, ...]  # the operators besides u, as written
    names: tuple[str, ...]
    heldout_error: float
    direct_error: float
    relative_cost: float
    reward: float  # accuracy + speed of the reward, without its format part
    block_sizes: tuple[int, ...]  # of the fit levels, then of the held-out level
    residual: np.ndarray  # (n_fit,) the fraction of the step of u the basis leaves unexplained at each fit level
    operators: list[OperatorFlow] = field(default_factory=list)
    candidates: list[tuple[str, float]] = field(default_factory=list)  # (monomial, relevance) at the coarsest fit level

    def text(self) -> str:
        b = self.block_sizes
        lines = [
            (f"Basis [{', '.join(('u', *self.texts))}]: held-out error {self.heldout_error:.4f} at b={b[-1]} with the "
             f"extrapolated kernels ({self.direct_error:.4f} with kernels fitted there), cost "
             f"x{self.relative_cost:.2f} of u alone, reward {self.reward:+.3f}."),
            "  unexplained step of u: " + ", ".join(f"{r:.3f} (b={s})" for r, s in zip(self.residual, b)),
        ]
        if self.operators:
            scales = " -> ".join(f"b={s}" for s in b)
            lines.append(f"  RG flow of the share of the step of u each operator carries ({scales}, the last one "
                         f"extrapolated):")
            for op in self.operators:
                trend = "grows" if op.trend > 0.5 else "shrinks" if op.trend < -0.5 else "about constant"
                lines.append(f"    {op.text}: " + " -> ".join(f"{s:.3f}" for s in op.share)
                             + f" ({trend}); without it the held-out error is x{op.without:.2f}")
            important = [op for op in self.operators if op.important]
            harmful = [op for op in self.operators if op.harmful]
            negligible = [op for op in self.operators if not (op.important or op.harmful)]
            if important:
                lines.append(f"  important at b={b[-1]}: " + ", ".join(op.text for op in important))
            if harmful:
                lines.append(f"  harmful at b={b[-1]} (the error is lower without it): "
                             + ", ".join(op.text for op in harmful))
            if negligible:
                lines.append(f"  negligible at b={b[-1]} (dropping one changes the error by less than "
                             f"{IMPORTANT - 1:.0%} and makes the closure cheaper): "
                             + ", ".join(op.text for op in negligible))
        if self.candidates:
            lines.append(f"  monomials outside the basis that best explain what it leaves at b={b[-2]}: "
                         + ", ".join(f"{text} {r:.3f}" for text, r in self.candidates))
        return "\n".join(lines)


def operator_shares(env: KernelEnvironment, ops: list[SymbolicOperator], kernels: np.ndarray, level: int) -> np.ndarray:
    """The share of the step of u carried by each operator of ops (the observables after u) with the given kernels
    (n_lags, K, M, M) at a level, on the training set: (len(ops),)."""
    g = env.basis_modes(ops, level)  # (batch, time, K, M)
    h = kernels.shape[0]
    windows = sliding_window_view(g[:, :-1], h, axis=1)  # (batch, time - h, K, M, h): g(s + w), w = 0 .. h-1
    # Omega^(l) acts on g(t - l): kernels reversed in lag line up with the window, oldest state first
    terms = np.einsum("lkj,bskjl->bskj", kernels[::-1, :, 0, :], windows, optimize=True)  # (batch, time - h, K, M)
    step = np.linalg.norm(g[:, h:, :, 0] - g[:, h - 1:-1, :, 0])
    return np.sqrt(np.sum(np.abs(terms[..., 1:]) ** 2, axis=(0, 1, 2))) / (step + 1e-30)


def flow_report(env: KernelEnvironment, ops: list[SymbolicOperator], config: RewardConfig = RewardConfig(),
                candidates: int = 5) -> FlowReport:
    ops = env.prepare(ops)
    evaluation = env.evaluate(ops)
    reward = score_basis(env, ops, config)
    levels = env.levels
    shares = [operator_shares(env, ops, env.kernels(ops, n).kernels, n) for n in levels.fit]
    shares.append(operator_shares(env, ops, evaluation.extrapolated, levels.heldout))
    shares = np.stack(shares, axis=1) if ops else np.zeros((0, len(levels.fit) + 1))  # (p, n_fit + 1)
    operators = [
        OperatorFlow(op.text, op.name, shares[i],
                     env.evaluate([o for o in ops if o is not op]).heldout_error / evaluation.heldout_error)
        for i, op in enumerate(ops)
    ]
    ranked: list[tuple[str, float]] = []
    if candidates:
        names = {op.name for op in ops}
        pool = [op for op in (SymbolicOperator(t) for t in all_monomials()) if op.degree > 1 and op.name not in names]
        relevance, _, _ = env.candidate_features(ops, pool, levels.fit[-1])
        ranked = [(op.text, float(r)) for r, op in sorted(zip(relevance, pool), key=lambda x: -x[0])[:candidates]]
    return FlowReport(
        texts=tuple(op.text for op in ops),
        names=tuple(op.name for op in ops),
        heldout_error=evaluation.heldout_error,
        direct_error=evaluation.direct_error,
        relative_cost=reward.relative_cost,
        reward=reward.accuracy + reward.speed,
        block_sizes=tuple(2**n for n in (*levels.fit, levels.heldout)),
        residual=evaluation.residual,
        operators=operators,
        candidates=ranked,
    )
