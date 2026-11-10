"""
reward = accuracy + speed + format
       = −log(error at the prediction scale)  −  speed_weight · log(relative cost)  −  penalties

A reply is read before anything is evaluated: lines that are not operators, operators that repeat an observable of the
basis, and operators that would make a step of the effective dynamics cost more than max_cost times a step of u alone (too slow
to evaluate) are taken out of the basis, each at a penalty. Only then is the basis evaluated and its reward computed.

"relative cost" measures how expensive one step of the effective dynamics with this basis is, compared with u alone (from speed.py).
"""

import math
import re
from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np

from core.operators.symbolic import OperatorSyntaxError, SymbolicOperator, is_x_derivative
from posttrain.speed import relative_cost
from posttrain.tasks import REWARD_SEED, Task, environment
from core.evaluation.kernels import KernelEnvironment

_BLOCK = re.compile(r"<basis>(.*?)</basis>", re.DOTALL)
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


MAX_OPS = 8  # a safety cap on the operators of a basis, besides u: each one enlarges and slows the kernel fit
# the most a step of the effective dynamics may cost, relative to u alone (speed.relative_cost), for a basis to be evaluated: the
# bases of the finite pool cost at most ~x15; an operator of 165 terms costs ~x64 and takes ~4 s to score
MAX_COST = 50.0


@dataclass(frozen=True)
class RewardConfig:
    speed_weight: float = 0.1
    invalid_penalty: float = 0.5  # per line that is not a usable operator, or too costly to evaluate
    duplicate_penalty: float = 0.5  # per operator that repeats an observable of the basis (u, linear, an x-derivative)
    missing_penalty: float = 1.0  # no <basis> block
    max_ops: int = MAX_OPS
    max_cost: float = MAX_COST  # relative cost of a step of the effective dynamics above which an operator is too costly to evaluate
    repeat_penalty: float = 0.5  # in the discovery loop, a basis already tried in the session: it brings nothing new
    reader: str = "basis"  # how a reply is turned into a basis, a key of READERS
    objective: str = "prediction"  # the error the accuracy is -log of, a key of OBJECTIVES


@dataclass
class Basis:
    ops: list[SymbolicOperator] = field(default_factory=list)
    invalid: list[str] = field(default_factory=list)  # lines that are not usable operators, with the reason
    dropped: list[str] = field(default_factory=list)  # lines that repeat an observable of the basis, with the reason
    found: bool = True  # whether the reply has a <basis> block


def _duplicate(op: SymbolicOperator, ops: list[SymbolicOperator]) -> str | None:
    """Why op adds no observable to u and ops, if it does not."""
    if op.name == "u":
        return "u is always included"
    if op.degree == 1:
        return "linear in u, the same observable as u"
    for other in ops:
        if op.name == other.name:
            return f"repeats {other.text}"
        if is_x_derivative(op, other):
            return f"an x-derivative of {other.text}, the same observable"
    return None


def parse_basis(reply: str, max_ops: int = MAX_OPS, cost: Callable[[list[SymbolicOperator]], float] | None = None,
                max_cost: float = MAX_COST) -> Basis:
    """The basis of a reply. Of an operator and its x-derivatives only the cheapest stays, the operator itself.
    cost: the relative cost of a step of the effective dynamics with a basis (speed.relative_cost on an environment); an operator that
    would take it above max_cost is too costly to evaluate. None: no cost check (no environment at hand)."""
    blocks = _BLOCK.findall(reply)
    if not blocks:
        return Basis(found=False)
    basis = Basis()
    for line in blocks[-1].replace(";", "\n").splitlines():
        text = _BULLET.sub("", line).strip().strip("`").strip()
        if not text:
            continue
        try:
            op = SymbolicOperator(text)
        except OperatorSyntaxError as e:
            basis.invalid.append(f"{text!r}: {e}")
            continue
        if op.n_space > 1:  # the fields of the tasks are 1D; evaluating this one would raise
            basis.invalid.append(f"{text!r}: u depends on x alone")
            continue
        reason = _duplicate(op, basis.ops)
        if reason:
            basis.dropped.append(f"{text!r}: {reason}")
            continue
        replaced = [o for o in basis.ops if is_x_derivative(o, op)]
        kept = [o for o in basis.ops if o not in replaced] + [op]
        if len(kept) > max_ops:
            basis.invalid.append(f"{text!r}: more than {max_ops} operators")
            continue
        if cost is not None and (c := cost(kept)) > max_cost:
            basis.invalid.append(f"{text!r}: too costly to evaluate: a step of the effective dynamics would cost x{c:.0f} a step of "
                                 f"u alone, above x{max_cost:g}")
            continue
        for other in replaced:
            basis.dropped.append(f"{other.text!r}: an x-derivative of {text}, the same observable")
        basis.ops = kept
    return basis


@dataclass(frozen=True)
class Score:
    accuracy: float
    speed: float
    format: float
    names: tuple[str, ...]  # the operators of the basis, in the environment's canonical order
    error: float
    relative_cost: float

    @property
    def total(self) -> float:
        return self.accuracy + self.speed + self.format


def basis_score(names: tuple[str, ...], error: float, relative_cost: float, config: RewardConfig = RewardConfig()) -> Score:
    """The accuracy and speed of a basis from its error and relative cost, without format penalty."""
    return Score(
        accuracy=-math.log(error),
        speed=-config.speed_weight * math.log(relative_cost),
        format=0.0,
        names=names,
        error=error,
        relative_cost=relative_cost,
    )


# ---- hooks: what a reply proposes, and what a basis is judged on ----
# Each is a registry with today's behaviour under the default name of its RewardConfig field. A future design adds an
# entry and selects it in the config; the default entries stay as they are.

# A reader turns a reply into the basis that is scored. "basis": the reply writes the basis itself. Future: the reply
# writes the features of a family of operators, and the reader runs greedy inside that family (search.family_pool).
Reader = Callable[[str, "RewardConfig", KernelEnvironment], Basis]
READERS: dict[str, Reader] = {"basis": lambda reply, config, env: parse_basis(
    reply, config.max_ops, None if env is None else (lambda ops: relative_cost(ops, env)), config.max_cost)}

# An objective is the error of a basis that the accuracy is -log of; lower is better. "prediction": the prediction
# error of the effective dynamics of u at the prediction scale, with extrapolated kernels. "leak": how far the span of (u, ops) is from invariant, the mean
# over its observables of the share of their step on the scoring trajectories the extrapolated kernels miss (KernelEnvironment.leak).
Objective = Callable[[KernelEnvironment, list[SymbolicOperator]], float]
OBJECTIVES: dict[str, Objective] = {
    "prediction": lambda env, ops: env.evaluate(ops).error,
    "leak": lambda env, ops: float(np.mean(env.leak(ops).rows)),
}


def read_reply(reply: str, config: "RewardConfig", env: KernelEnvironment) -> Basis:
    return READERS[config.reader](reply, config, env)


def score_basis(env: KernelEnvironment, ops: list[SymbolicOperator], config: RewardConfig = RewardConfig()) -> Score:
    """The accuracy and speed of a basis on an environment, without format penalty."""
    evaluation = env.evaluate(ops)
    error = evaluation.error if config.objective == "prediction" else OBJECTIVES[config.objective](env, ops)
    return basis_score(evaluation.names, error, relative_cost(env.prepare(ops), env), config)


def score(task: "Task | str", reply: str, config: RewardConfig = RewardConfig(), seed: int = REWARD_SEED) -> Score:
    """The reward of a reply on a task (a Task, or the name of one of the bank), on the ensemble of a seed."""
    env = environment(task, seed)
    basis = read_reply(reply, config, env)
    if not basis.found:
        return Score(-math.log(env.max_error), 0.0, -config.missing_penalty, (), env.max_error, 1.0)
    return replace(score_basis(env, basis.ops, config), format=penalty(basis, config))


def penalty(basis: Basis, config: RewardConfig = RewardConfig()) -> float:
    """The format part of the reward of a reply with a <basis> block: its unusable lines and its repeated operators."""
    return -config.invalid_penalty * len(basis.invalid) - config.duplicate_penalty * len(basis.dropped)

