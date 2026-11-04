"""
reward = accuracy + speed + format
       = −log(held-out error)  −  speed_weight · log(relative cost)  −  penalties

"relative cost" measures how expensive one closure step with this basis is, compared with u alone (from speed.py).
"""

import math
import re
from dataclasses import dataclass, field, replace

from core.operators.symbolic import OperatorSyntaxError, SymbolicOperator
from posttrain.speed import relative_cost
from posttrain.tasks import DEMO, REWARD_SEED, EnvSize, environment, task
from core.evaluation.kernels import KernelEnvironment

_BLOCK = re.compile(r"<basis>(.*?)</basis>", re.DOTALL)
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


@dataclass(frozen=True)
class RewardConfig:
    speed_weight: float = 0.1
    invalid_penalty: float = 0.5  # per line that is not a usable operator
    missing_penalty: float = 1.0  # no <basis> block
    max_ops: int = 4


@dataclass
class Basis:
    ops: list[SymbolicOperator] = field(default_factory=list)
    invalid: list[str] = field(default_factory=list)  # lines that are not usable operators, with the reason
    found: bool = True  # whether the reply has a <basis> block


def parse_basis(reply: str, max_ops: int = 4) -> Basis:
    blocks = _BLOCK.findall(reply)
    if not blocks:
        return Basis(found=False)
    basis, names = Basis(), {"u"}
    for line in blocks[-1].replace(";", "\n").splitlines():
        text = _BULLET.sub("", line).strip().strip("`").strip()
        if not text:
            continue
        try:
            op = SymbolicOperator(text)
        except OperatorSyntaxError as e:
            basis.invalid.append(f"{text!r}: {e}")
            continue
        if op.name in names:
            basis.invalid.append(f"{text!r}: " + ("u is always included" if op.name == "u" else f"repeats {op.name}"))
        elif len(basis.ops) == max_ops:
            basis.invalid.append(f"{text!r}: more than {max_ops} operators")
        else:
            basis.ops.append(op)
            names.add(op.name)
    return basis


@dataclass(frozen=True)
class Score:
    accuracy: float
    speed: float
    format: float
    names: tuple[str, ...]  # the operators of the basis, in the environment's canonical order
    heldout_error: float
    relative_cost: float

    @property
    def total(self) -> float:
        return self.accuracy + self.speed + self.format


def basis_score(names: tuple[str, ...], heldout_error: float, relative_cost: float, config: RewardConfig = RewardConfig()) -> Score:
    """The accuracy and speed of a basis from its held-out error and relative cost, without format penalty."""
    return Score(
        accuracy=-math.log(heldout_error),
        speed=-config.speed_weight * math.log(relative_cost),
        format=0.0,
        names=names,
        heldout_error=heldout_error,
        relative_cost=relative_cost,
    )


def score_basis(env: KernelEnvironment, ops: list[SymbolicOperator], config: RewardConfig = RewardConfig()) -> Score:
    """The accuracy and speed of a basis on an environment, without format penalty."""
    evaluation = env.evaluate(ops)
    return basis_score(evaluation.names, evaluation.heldout_error, relative_cost(env.prepare(ops), env), config)


def score(task_name: str, reply: str, config: RewardConfig = RewardConfig(), size: EnvSize = DEMO, seed: int = REWARD_SEED) -> Score:
    env = environment(task(task_name), size, seed)
    basis = parse_basis(reply, config.max_ops)
    if not basis.found:
        return Score(-math.log(env.max_error), 0.0, -config.missing_penalty, (), env.max_error, 1.0)
    return replace(score_basis(env, basis.ops, config), format=-config.invalid_penalty * len(basis.invalid))


def _text(completion) -> str:
    """The text of a completion, plain or conversational ([{"role": "assistant", "content": ...}])."""
    if isinstance(completion, str):
        return completion
    return "".join(message["content"] for message in completion if message.get("role", "assistant") == "assistant")


def reward_functions(config: RewardConfig = RewardConfig(), size: EnvSize = DEMO, seed: int = REWARD_SEED) -> list:
    """accuracy, speed and format rewards for TRL's GRPOTrainer; TRL logs each under the function's name."""
    cache: dict[tuple[str, str], Score] = {}

    def scores(completions, task) -> list[Score]:
        out = []
        for completion, name in zip(completions, task):
            key = (name, _text(completion))
            if key not in cache:
                cache[key] = score(name, key[1], config, size, seed)
            out.append(cache[key])
        return out

    def accuracy_reward(prompts, completions, task, **kwargs) -> list[float]:
        return [s.accuracy for s in scores(completions, task)]

    def speed_reward(prompts, completions, task, **kwargs) -> list[float]:
        return [s.speed for s in scores(completions, task)]

    def format_reward(prompts, completions, task, **kwargs) -> list[float]:
        return [s.format for s in scores(completions, task)]

    return [accuracy_reward, speed_reward, format_reward]
