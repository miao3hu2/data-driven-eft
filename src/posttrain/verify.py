from dataclasses import dataclass, field

from posttrain.reward import RewardConfig, Score, read_reply, score_basis
from posttrain.search import SearchConfig, SearchResult, greedy, known, local_search, nonlinear_monomials, reward
from posttrain.speed import measure_speedup
from posttrain.tasks import REWARD_SEED, VERIFY_SEED, Task, as_task, environment

__all__ = ["CheckConfig", "Verification", "baseline", "verify", "nonlinear_monomials"]


@dataclass(frozen=True)
class CheckConfig:
    local_optimum: bool = True  # no single move (drop, add, swap one operator) may raise the reward by more than tol
    beat_baseline: bool = True  # the reward may not be below that of greedy forward selection by more than tol
    best_known: bool = True  # no known basis (rivals, moves of the searches) may have a reward higher by more than tol
    tol: float = 1e-6


@dataclass
class Verification:
    task: str
    seed: int  # the ensemble of the check
    proposal: Score
    best: Score  # the best known basis, the proposal included, by accuracy + speed
    rank: int
    n_known: int  # known bases besides the proposal
    dominated_by: list[Score]  # the known bases at least as accurate and as cheap as the proposal, better in one
    baseline: SearchResult  # greedy forward selection
    local: SearchResult  # local search from the proposal
    invalid: list[str]  # lines of the reply that are not usable operators, see reward.parse_basis
    found: bool  # whether the reply has a <basis> block; without one the proposal is scored as u alone
    check: CheckConfig = field(default_factory=CheckConfig)
    n_evaluated: int = 0  # bases this verification evaluated, not evaluated on this ensemble before
    effective_seconds: float | None = None  # wall time of the effective dynamics per snapshot interval
    simulation_seconds: float | None = None
    rivals: list[Score] = field(default_factory=list)  # the best bases of the rival ensembles, on this ensemble

    @property
    def pareto_optimal(self) -> bool | None:
        return None if not self.found else not self.dominated_by

    @property
    def local_gain(self) -> float:
        return self.local.gain

    @property
    def beats_baseline(self) -> bool:
        return reward(self.proposal) >= reward(self.baseline.best) - self.check.tol

    @property
    def failures(self) -> list[str]:
        """Why the proposal does not pass, worded for the model: what fails, not which basis would pass."""
        out = []
        if not self.found:
            out.append("the reply has no <basis> block")
        if self.check.local_optimum and self.local_gain > self.check.tol:
            out.append("it is not a local optimum: dropping, adding or swapping a single operator gives a better basis")
        if self.check.beat_baseline and not self.beats_baseline:
            out.append("a greedy search that adds one operator at a time, starting from u alone, finds a better basis")
        if self.check.best_known and reward(self.best) > reward(self.proposal) + self.check.tol and not out:
            out.append("a better basis is already known on this data")  # said only when nothing above explains it
        return out

    @property
    def passed(self) -> bool:
        return not self.failures

    @property
    def speedup(self) -> float | None:
        if self.effective_seconds is None or self.simulation_seconds is None:
            return None
        return self.simulation_seconds / self.effective_seconds

    def summary(self) -> dict:
        """The verification as JSON-serializable values."""
        return dict(task=self.task, 
                    seed=self.seed, 
                    passed=self.passed, 
                    failures=self.failures, 
                    found=self.found,
                    invalid=self.invalid, 
                    basis=list(self.proposal.names), error=self.proposal.error,
                    relative_cost=self.proposal.relative_cost, 
                    reward=reward(self.proposal), 
                    rank=self.rank,
                    n_known=self.n_known, 
                    pareto_optimal=self.pareto_optimal, 
                    best=list(self.best.names),
                    best_reward=reward(self.best), 
                    local_gain=self.local_gain, 
                    local_best=list(self.local.best.names),
                    baseline=list(self.baseline.best.names), 
                    baseline_reward=reward(self.baseline.best),
                    beats_baseline=self.beats_baseline, 
                    n_evaluated=self.n_evaluated, 
                    speedup=self.speedup)

    def report(self) -> str:
        p, b, g, loc = self.proposal, self.best, self.baseline.best, self.local
        pareto = {None: "no proposal", 
                  True: "Pareto optimal among them",
                  False: f"dominated by {len(self.dominated_by)} of them"}[self.pareto_optimal]
        lines = [
            f"{self.task}: proposal {_basis(p)}: error {p.error:.4f}, cost x{p.relative_cost:.2f} "
            f"of u alone, reward {reward(p):+.3f}; rank {self.rank} of {self.n_known + 1} known bases, {pareto}",
            "  check: " + ("passed" if self.passed else "failed, " + "; ".join(self.failures)),
            ("  local optimum: no single move improves it" if loc.gain <= self.check.tol else
             f"  not a local optimum: {len(loc.path) - 1} moves reach {_basis(loc.best)}, reward "
             f"{reward(loc.best):+.3f} ({loc.gain:+.3f})"),
            (f"  greedy baseline {_basis(g)}: reward {reward(g):+.3f}, "
             + ("beaten or matched" if self.beats_baseline else "better than the proposal")),
            (f"  best known: {_basis(b)}: error {b.error:.4f}, cost x{b.relative_cost:.2f}, "
             f"reward {reward(b):+.3f}"),
        ]
        if self.found and self.invalid:
            lines.insert(1, f"  {len(self.invalid)} unusable lines, ignored: " + "; ".join(self.invalid))
        effective, simulation = self.effective_seconds, self.simulation_seconds
        if effective is not None and simulation is not None:
            lines.append(f"  effective dynamics {effective * 1e3:.2f} ms per snapshot, direct simulation {simulation * 1e3:.1f} ms: "
                         f"{simulation / effective:.0f} times faster")
        return "\n".join(lines)


def _basis(s: Score) -> str:
    return "[" + ", ".join(("u", *s.names)) + "]"


def baseline(task: "Task | str", config: RewardConfig = RewardConfig(), seed: int = VERIFY_SEED, search: SearchConfig = SearchConfig()) -> SearchResult:
    """Greedy forward selection on the ensemble of the check; the loop computes it once per task and reuses it."""
    return greedy(environment(task, seed), config, search)


def verify(task: "Task | str",
           reply: str, 
           config: RewardConfig = RewardConfig(), 
           seed: int = VERIFY_SEED, 
           measure: bool = True, 
           search: SearchConfig = SearchConfig(),
           check: CheckConfig = CheckConfig(), 
           greedy_baseline: SearchResult | None = None,
           rival_seeds: tuple[int, ...] = (REWARD_SEED,)
           ) -> Verification:
    t = as_task(task)
    env = environment(t, seed)
    before = env.n_evaluations
    basis = read_reply(reply, config, env)
    proposal = score_basis(env, basis.ops, config)
    rivals: dict[tuple[str, ...], Score] = {}
    for rival_seed in rival_seeds:  # a repeated seed offers the same bases again: skipped below, at no cost
        if rival_seed == seed:
            continue
        other = environment(t, rival_seed)
        for s in known(other, config)[:search.rivals]:
            if s.names not in rivals:
                rivals[s.names] = score_basis(env, [other.operators[name] for name in s.names], config)
    if greedy_baseline is None:
        greedy_baseline = greedy(env, config, search)
    elif greedy_baseline.start != score_basis(env, [], config):
        raise ValueError(f"greedy_baseline was not computed on this ensemble (seed {seed}): its start, u alone, has "
                         f"error {greedy_baseline.start.error:.6g} instead of "
                         f"{env.evaluate([]).error:.6g}")
    local = local_search(env, basis.ops, config, search)
    known_here = {**rivals, **greedy_baseline.scored, **local.scored}
    others = [s for names, s in known_here.items() if names != proposal.names]
    verification = Verification(
        task=t.name,
        seed=seed,
        proposal=proposal,
        best=max([proposal, *others], key=reward),
        rank=1 + sum(reward(s) > reward(proposal) for s in others),
        n_known=len(others),
        dominated_by=[s for s in others if s.error <= proposal.error
                      and s.relative_cost <= proposal.relative_cost
                      and (s.error < proposal.error or s.relative_cost < proposal.relative_cost)],
        baseline=greedy_baseline,
        local=local,
        invalid=basis.invalid,
        found=basis.found,
        check=check,
        n_evaluated=env.n_evaluations - before,
        rivals=list(rivals.values()),
    )
    if measure:
        verification.effective_seconds, verification.simulation_seconds = measure_speedup(env, basis.ops)
    return verification

