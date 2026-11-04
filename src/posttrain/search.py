from dataclasses import dataclass, field
from typing import Callable, Literal

from core.evaluation.kernels import KernelEnvironment
from core.operators.symbolic import SymbolicOperator, all_monomials
from posttrain.reward import RewardConfig, Score, score_basis

Pool = Callable[[list[SymbolicOperator]], list[SymbolicOperator]]
Space = Literal["finite", "infinite"]


@dataclass(frozen=True)
class SearchConfig:
    space: Space = "finite"  # where the operators of the searches come from, see SPACES
    screen: int = 5  # operators of the pool evaluated in full per step, the most relevant ones
    max_steps: int = 3  # moves of the local search
    rivals: int = 5  # best bases the training evaluated that the check evaluates again, on its own ensemble
    max_ops: int | None = None  # largest basis, RewardConfig.max_ops if None


def nonlinear_monomials() -> list[SymbolicOperator]:
    """All of the nonlinear monomials of the DSL."""
    return [op for op in (SymbolicOperator(text) for text in all_monomials()) if op.degree > 1]


def monomial_pool(ops: list[SymbolicOperator]) -> list[SymbolicOperator]:
    """The nonlinear monomials of the DSL outside the basis."""
    names = {op.name for op in ops}
    return [op for op in nonlinear_monomials() if op.name not in names]


SPACES: dict[str, Pool] = {"finite": monomial_pool}  # "infinite": a generator of operators, to come in the future


def pool_of(space: str) -> Pool:
    if space in SPACES:
        return SPACES[space]
    if space == "infinite":
        raise NotImplementedError("the infinite space of operators has no generator yet: add a Pool, a function "
                                  "from a basis to the operators that can be added to it, as SPACES['infinite']")
    raise ValueError(f"unknown space {space!r}, one of {sorted(SPACES)} or 'infinite'")


def reward(s: Score) -> float:
    return s.accuracy + s.speed


@dataclass
class SearchResult:
    start: Score
    best: Score
    path: list[Score] = field(default_factory=list)  # the bases moved through, start first, best last
    scored: dict[tuple[str, ...], Score] = field(default_factory=dict)  # every basis the search scored, by names

    @property
    def n_scored(self) -> int:
        return len(self.scored)

    @property
    def gain(self) -> float:
        return reward(self.best) - reward(self.start)


def known(env: KernelEnvironment, config: RewardConfig = RewardConfig()) -> list[Score]:
    """Every basis the environment has evaluated, best first by accuracy + speed."""
    out = [score_basis(env, [env.operators[name] for name in e.names], config) for e in env.history]
    return sorted(out, key=lambda s: -reward(s))


def screened(env: KernelEnvironment, ops: list[SymbolicOperator], pool: Pool, n: int) -> list[SymbolicOperator]:
    """The n operators of the pool that could explain most of what the basis leaves in u, most relevant first."""
    candidates = pool(ops)
    if not candidates or n <= 0:
        return []
    if len(candidates) <= n:
        return candidates
    relevance, _, _ = env.candidate_features(env.prepare(ops), candidates, env.levels.fit[-1])
    ranked = sorted(zip(relevance, range(len(candidates))), key=lambda r: -r[0])
    return [candidates[i] for _, i in ranked[:n]]


def _max_ops(config: RewardConfig, search: SearchConfig) -> int:
    return config.max_ops if search.max_ops is None else search.max_ops


def _climb(env: KernelEnvironment, ops: list[SymbolicOperator], moves: Callable, config: RewardConfig,max_steps: int) -> SearchResult:
    """Moves to the best of moves(ops) while it raises the reward, for at most max_steps moves."""
    ops = env.prepare(ops)
    current = score_basis(env, ops, config)
    result = SearchResult(current, current, [current], {current.names: current})
    for _ in range(max_steps):
        scored = [(score_basis(env, m, config), m) for m in moves(ops)]
        result.scored.update((s.names, s) for s, _ in scored)
        if not scored:
            break
        s, m = max(scored, key=lambda x: reward(x[0]))
        if reward(s) <= reward(current):
            break
        ops, current = env.prepare(m), s
        result.path.append(s)
    result.best = current
    return result


def greedy(env: KernelEnvironment, config: RewardConfig = RewardConfig(), search: SearchConfig = SearchConfig()) -> SearchResult:
    """Forward selection from u alone."""
    pool, n = pool_of(search.space), _max_ops(config, search)
    return _climb(env, [], lambda ops: [ops + [c] for c in screened(env, ops, pool, search.screen)]
                  if len(ops) < n else [], config, n)


def neighbours(env: KernelEnvironment, ops: list[SymbolicOperator], config: RewardConfig = RewardConfig(),search: SearchConfig = SearchConfig()) -> list[list[SymbolicOperator]]:
    """The bases one move away: drop one operator, add one of the screened, or swap one for one of the screened."""
    added = screened(env, ops, pool_of(search.space), search.screen)
    moves = [ops[:i] + ops[i + 1:] for i in range(len(ops))]
    if len(ops) < _max_ops(config, search):
        moves += [ops + [c] for c in added]
    moves += [ops[:i] + [c] + ops[i + 1:] for i in range(len(ops)) for c in added]
    return moves


def local_search(env: KernelEnvironment, ops: list[SymbolicOperator], config: RewardConfig = RewardConfig(),
                 search: SearchConfig = SearchConfig()) -> SearchResult:
    """Hill climbing from a basis, by the best single move."""
    return _climb(env, ops, lambda b: neighbours(env, b, config, search), config, search.max_steps)
