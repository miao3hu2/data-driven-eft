import dataclasses
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from core.operators.symbolic import SymbolicOperator
from posttrain.feedback import FlowReport, flow_report
from posttrain.policy import Group, Proposer
from posttrain.prompts import SYSTEM, Hints, diagnostics
from posttrain.reward import RewardConfig, Score, parse_basis, score
from posttrain.search import SearchConfig, SearchResult, greedy, reward
from posttrain.tasks import FINAL_SEED, REWARD_SEED, SIZES, VERIFY_SEED, EnvSize, Task, environment, split
from posttrain.verify import CheckConfig, Verification, baseline, verify

SYSTEM_LOOP = SYSTEM + """

You work in turns. After each proposal you get the result: the error of the closure at the coarsest scale, its cost,
its reward (accuracy + speed, higher is better) and the RG flow of the basis: the share of the step of u each operator
carries from fine to coarse scales, how much the error grows without it, which operators are important at the coarse
scale, which are negligible there and which are harmful (the closure is better without them), and which monomials
explain what the basis still misses. Use it: keep the important operators, drop the negligible ones (they only cost)
and the harmful ones, and try the missing ones. Your best basis is also checked on independent data: it passes if no
single change (dropping, adding or swapping one operator) improves it, a simple greedy search does not beat it, and
no better basis is known there.
You keep proposing until it passes, and you are told why it failed."""


@dataclass
class Attempt:
    turn: int
    reply: str
    score: Score
    report: FlowReport

    @property
    def total(self) -> float:
        return self.score.total

    def line(self) -> str:
        return (f"[{', '.join(('u', *self.report.texts))}]: error {self.score.heldout_error:.4f}, "
                f"cost x{self.score.relative_cost:.2f}, reward {reward(self.score):+.3f}")


@dataclass
class Session:
    task: str
    chosen: list[Attempt] = field(default_factory=list)  # per turn, the reply carried forward
    groups: list[Group] = field(default_factory=list)  # per turn, every reply and its reward
    checks: list[Verification] = field(default_factory=list)  # per turn, the check of the best basis so far

    @property
    def passed(self) -> bool:
        return bool(self.checks) and self.checks[-1].passed

    @property
    def best(self) -> Attempt:
        return max(self.chosen, key=lambda a: a.total)

    def summary(self) -> dict:
        return dict(task=self.task, 
                    turns=[
                        dict(basis=list(a.score.names), 
                            reward=a.total,
                            heldout_error=a.score.heldout_error, 
                            cost=a.score.relative_cost)
                        for a in self.chosen
                        ],
                    best=list(self.best.score.names), 
                    best_reward=self.best.total, 
                    first_reward=self.chosen[0].total,
                    passed=self.passed, 
                    n_turns=len(self.chosen))


def session_messages(t: Task, size: EnvSize, history: list[Attempt], hints: Hints = "stats", system_role: bool = True, max_history: int = 6, check: Verification | None = None) -> list[dict]:
    """The prompt of a turn; check: the last check of the best basis, said when it failed."""
    parts = [diagnostics(environment(t, size), hints)]
    if history:
        tried = {}
        for a in sorted(history, key=lambda a: -a.total):
            tried.setdefault(a.score.names, a)  # the same basis written twice is listed once
        parts.append("Bases you have tried, best first:\n" + "\n".join(
            f"- {a.line()}" for a in list(tried.values())[:max_history]))
        best, last = max(history, key=lambda a: a.total), history[-1]
        parts.append("RG flow of your best basis:\n" + best.report.text())
        if last.score.names != best.score.names:
            parts.append("RG flow of your latest basis:\n" + last.report.text())
        if check is not None and not check.passed:
            parts.append("Check of your best basis on independent data: failed, " + "; ".join(check.failures) + ".")
        parts.append("Propose a better basis: more accurate at the coarsest scale, or as accurate and cheaper.")
    else:
        parts.append("Propose the basis.")
    user = "\n\n".join(parts)
    if system_role:
        return [{"role": "system", "content": SYSTEM_LOOP}, {"role": "user", "content": user}]
    return [{"role": "user", "content": SYSTEM_LOOP + "\n\n" + user}]


@dataclass
class LoopConfig:
    rounds: int = 12  # each round runs a session on every training task
    max_turns: int = 4  # proposals per session at most, each after the feedback on the previous one; a session
    # ends earlier when its best basis passes the check
    group: int = 8  # replies sampled per turn, the GRPO group
    update_every: int = 2  # rounds between post-training updates
    size: str = "demo"  # environment size, a key of tasks.SIZES
    hints: Hints = "stats"  # what the diagnostics show; the RG flow feedback lists the missing monomials anyway
    candidates: int = 5  # monomials outside the basis listed in the feedback, 0 for none
    system_role: bool = True
    reference: bool = True  # the greedy baseline on the reward ensemble as the reference of the regret; False skips it
    search: SearchConfig = field(default_factory=SearchConfig)  # of the reference and of the check, and its space
    check: CheckConfig = field(default_factory=CheckConfig)  # what a basis must pass to end a session
    check_seed: int = VERIFY_SEED  # the ensemble of the check, whose verdict the model is told
    final_seed: int = FINAL_SEED  # the ensemble of the final verification, never seen by the loop
    output_dir: str = "experiments/06_llm_posttrain/runs/loop"
    reward: RewardConfig = field(default_factory=RewardConfig)


def attempt(t: Task, reply: str, config: LoopConfig, turn: int = 0, scored: Score | None = None) -> Attempt:
    """A reply scored by the reward, with the RG flow report of its basis: what the model is told about it. scored: the score of the reply, if already computed."""
    size = SIZES[config.size]
    scored = score(t.name, reply, config.reward, size) if scored is None else scored
    basis = parse_basis(reply, config.reward.max_ops)
    report = flow_report(environment(t, size, REWARD_SEED), basis.ops, config.reward, config.candidates)
    return Attempt(turn, reply, scored, report)


def _check(t: Task, reply: str, config: LoopConfig, seed: int, **options) -> Verification:
    """verify.verify on the ensemble of `seed`, with the settings of the loop."""
    return verify(t.name, reply, config.reward, SIZES[config.size], seed, search=config.search, check=config.check, **options)


def run_session(policy: Proposer, t: Task, config: LoopConfig, greedy: bool = False, round: int = 0, greedy_baseline: SearchResult | None = None) -> Session:
    """Proposals on one task until the best basis passes the check, at most config.max_turns of them. greedy_baseline: the check's baseline for the task, computed once by the loop."""
    size = SIZES[config.size]
    session, checked = Session(t.name), {}
    for turn in range(config.max_turns):
        prompt = session_messages(t, size, session.chosen, config.hints, config.system_role, check=session.checks[-1] if session.checks else None)
        completions = policy.generate(prompt, 1 if greedy else config.group, greedy=greedy)
        scores = [score(t.name, c.text, config.reward, size) for c in completions]
        session.groups.append(Group(t.name, prompt, completions, [s.total for s in scores], round, turn))
        j = max(range(len(scores)), key=lambda i: scores[i].total)
        session.chosen.append(attempt(t, completions[j].text, config, turn, scores[j]))
        reply = session.best.reply
        if reply not in checked:  # the best basis may not change from one turn to the next
            checked[reply] = _check(t, reply, config, config.check_seed, measure=False, greedy_baseline=greedy_baseline)
        session.checks.append(checked[reply])
        if session.passed:
            break
    return session


def greedy_reference(t: Task, config: LoopConfig) -> Score:
    """The reference of the regret: the basis greedy forward selection finds on the reward ensemble."""
    return greedy(environment(t, SIZES[config.size], REWARD_SEED), config.reward, config.search).best


class DiscoveryLoop:
    def __init__(self, policy: Proposer, config: LoopConfig | None = None, train: list[Task] | None = None, test: list[Task] | None = None, log=print):
        self.policy, self.config, self.log = policy, LoopConfig() if config is None else config, log
        self.train = split("train") if train is None else train
        self.test = split("test") if test is None else test
        self.tasks = self.train + self.test
        self.out = Path(self.config.output_dir)
        if self.config.final_seed in (REWARD_SEED, self.config.check_seed):
            raise ValueError(f"final_seed {self.config.final_seed} must differ from the reward seed {REWARD_SEED} and from check_seed {self.config.check_seed}: the final verification needs data the loop has not seen")
        self.baselines: dict[str, SearchResult] = {}  # the check's greedy baseline per task, computed once
        self.history: list[dict] = []  # one entry per round and per evaluation
        self.references: dict[str, Score] = {}
        self.final: list[Session] = []

    def regret(self, task: str, a: Attempt) -> float | None:
        """The reward of the reference basis minus the reward of the attempt; None without a reference."""
        ref = self.references.get(task)
        return None if ref is None else reward(ref) - a.total

    def baseline(self, t: Task) -> SearchResult:
        if t.name not in self.baselines:
            cfg = self.config
            self.baselines[t.name] = baseline(t.name, cfg.reward, SIZES[cfg.size], cfg.check_seed, cfg.search)
        return self.baselines[t.name]

    def evaluate(self, round: int) -> list[Session]:
        """A greedy session on every task with the current weights."""
        sessions = []
        for t in self.tasks:
            s = run_session(self.policy, t, self.config, greedy=True, round=round, greedy_baseline=self.baseline(t))
            sessions.append(s)
            entry = self._record("eval", round, t, s)
            n_turns = len(s.chosen)
            outcome = (f"passed in {n_turns} turns" if s.passed else f"not passed in {n_turns} turns: " + "; ".join(s.checks[-1].failures))
            lines = [f"  [eval {t.split:5s}] {t.name:16s} {outcome}"]
            for label, a, regret in (("first", s.chosen[0], entry["first_regret"]), ("best ", s.best, entry["best_regret"])):
                lines.append(f"  {'':30s} {label} {a.line()}" + ("" if regret is None else f" regret {regret:+.3f}"))
            self.log("\n".join(lines))
        return sessions

    def run(self) -> list[Verification]:
        cfg = self.config
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "loop_config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))
        if cfg.reference and not self.references:  # references set beforehand are kept
            self._find_references()
        self.log("[eval] before post-training")
        sessions = self.evaluate(0)
        buffer: list[Group] = []
        for r in range(1, cfg.rounds + 1):
            buffer += self._sample(r)
            if r % cfg.update_every == 0 or r == cfg.rounds:
                self._update(r, buffer)
                buffer = []  # the next groups come from the updated weights
                sessions = self.evaluate(r)
                self._save(sessions)
        self.final = sessions
        verifications = []
        for t, s in zip(self.tasks, sessions):
            # its own greedy baseline, on that seed; rivals from the training and from every check of the run
            v = _check(t, s.best.reply, cfg, cfg.final_seed, measure=True, rival_seeds=(REWARD_SEED, cfg.check_seed))
            verifications.append(v)
            self.log(f"[verify {t.split}, final ensemble] " + ("" if s.passed else "(the session did not pass the check) ") + v.report())
        self._save(sessions, verifications)
        return verifications

    def _find_references(self) -> None:
        t0 = time.time()
        for t in self.tasks:
            self.references[t.name] = greedy_reference(t, self.config)
        found = ", ".join(f"{name} {list(s.names)}" for name, s in self.references.items())
        self.log(f"[references] greedy forward selection ({time.time() - t0:.0f}s): " + found)

    def _sample(self, r: int) -> list[Group]:
        """A sampled session on every training task; returns their groups."""
        t0 = time.time()
        groups, entries = [], []
        for t in self.train:
            s = run_session(self.policy, t, self.config, round=r, greedy_baseline=self.baseline(t))
            groups += s.groups
            rewards = [x for g in s.groups for x in g.rewards]
            entries.append(self._record("sample", r, t, s, mean_reward=sum(rewards) / len(rewards)))
        mean = sum(e["mean_reward"] for e in entries) / len(entries)
        best = ", ".join(f"{e['task']} {e['best_reward']:+.3f}" for e in entries)
        passed = sum(e["passed"] for e in entries)
        self.log(f"[round {r}] mean reward of the samples {mean:+.3f}, best per task {best}, {passed}/{len(entries)} sessions passed the check ({time.time() - t0:.0f}s)")
        return groups

    def _update(self, r: int, groups: list[Group]) -> None:
        """Post-train the policy on the groups and save a checkpoint."""
        t0 = time.time()
        stats = self.policy.update(groups)
        self.history.append(dict(kind="update", round=r, **stats))
        self.log(f"[update after round {r}] " + ", ".join(f"{k} {v:.4g}" for k, v in stats.items())
                 + f" ({time.time() - t0:.0f}s)")
        self.policy.save(str(self.out / f"checkpoint-{r}"))

    def _record(self, kind: str, round: int, t: Task, s: Session, **extra) -> dict:
        """Add a session to the history, with the regret of its first and best proposals."""
        entry = dict(kind=kind, round=round, split=t.split, **s.summary(), **extra,
                     first_regret=self.regret(s.task, s.chosen[0]), best_regret=self.regret(s.task, s.best))
        self.history.append(entry)
        return entry

    def _save(self, sessions: list[Session], verifications: list[Verification] | None = None) -> None:
        (self.out / "history.json").write_text(json.dumps(self.history, indent=1))
        transcripts = [dict(task=s.task, 
                            turns=[
                                dict(prompt=g.prompt[-1]["content"], 
                                     reply=a.reply,
                                     basis=list(a.score.names), 
                                     reward=a.total)
                                for g, a in zip(s.groups, s.chosen)]) 
                        for s in sessions]
        (self.out / "last_eval_sessions.json").write_text(json.dumps(transcripts, indent=1))
        if verifications is not None:
            (self.out / "verification.json").write_text(json.dumps([
                dict(v.summary(), operators=list(s.best.report.texts), passed_check=s.passed)
                for s, v in zip(sessions, verifications)], indent=1))


def final_bases(output_dir: str | Path) -> dict[str, list[SymbolicOperator]]:
    """The basis the post-trained model ended with on every task of a finished loop, from its verification.json: {task: operators besides u}."""
    entries = json.loads((Path(output_dir) / "verification.json").read_text())
    missing = [e["task"] for e in entries if "operators" not in e]
    if missing:
        raise ValueError(f"{output_dir}/verification.json has no operators for {missing}: rerun the loop, or read the bases from last_eval_sessions.json")
    return {e["task"]: [SymbolicOperator(text) for text in e["operators"]] for e in entries}


def learning_curves(history: list[dict]) -> dict[str, dict[str, list[float]]]:
    """The evaluations of a loop (DiscoveryLoop.history) by round: for each split, the mean over its tasks of the
    regret of the first proposal and of the best one, the reward of the best one, the fraction of sessions that passed
    the check and their number of turns; and the weight change after each update. {"train": {"round": [...],
    "first_regret": [...], "best_regret": [...], "best_reward": [...], "passed": [...], "n_turns": [...]},
    "test": {...}, "updates": {"round": [...], "weight_change": [...]}}"""
    curves: dict[str, dict[str, list[float]]] = {}
    evals = [h for h in history if h["kind"] == "eval"]
    for part in ("train", "test"):
        rows = [h for h in evals if h["split"] == part]
        rounds = sorted({h["round"] for h in rows})
        curves[part] = {"round": rounds}
        for key in ("first_regret", "best_regret", "best_reward", "passed", "n_turns"):
            values = [[float(h[key]) for h in rows if h["round"] == r and h.get(key) is not None] for r in rounds]
            curves[part][key] = [sum(v) / len(v) if v else float("nan") for v in values]
    updates = [h for h in history if h["kind"] == "update"]
    curves["updates"] = {
        "round": [h["round"] for h in updates],
        "weight_change": [h.get("weight_change", float("nan")) for h in updates]
        }
    return curves
