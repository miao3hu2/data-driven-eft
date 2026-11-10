import dataclasses
import json
import random
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

from core.evaluation.kernels import KernelEnvironment
from core.operators.symbolic import SymbolicOperator
from posttrain.feedback import FlowReport, flow_report
from posttrain.policy import Group, Proposer
from posttrain.prompts import SYSTEM, Hints, diagnostics
from posttrain.reward import RewardConfig, Score, read_reply, score
from posttrain.search import SearchConfig, SearchResult, greedy, reward
from posttrain.tasks import FINAL_SEED, REWARD_SEED, VERIFY_SEED, Task, environment, release, split
from posttrain.verify import CheckConfig, Verification, baseline, verify

def system_loop(listed: bool = True) -> str:
    """The system prompt of the discovery loop. listed: whether the feedback lists the monomials that explain what a
    basis still misses (LoopConfig.candidates > 0); without the list the model is not told to expect one."""
    missing = (", and which monomials\nexplain what the basis still misses. Use it: keep the important operators, drop the negligible ones (they only cost)\n"
               "and the harmful ones, and try the missing ones." if listed else
               ".\nUse it: keep the important operators, drop the negligible ones (they only cost) and the harmful ones, and find\n"
               "operators for what the basis still misses.")
    return SYSTEM + """

You work in turns. After each proposal you get the result: the prediction error of its effective dynamics at the coarsest scale, its cost,
its reward (accuracy + speed, higher is better) and the RG flow of the basis: the share of the step of u each operator
carries from fine to coarse scales, how much the error grows without it, which operators are important at the coarse
scale, which are negligible there and which are harmful (the effective dynamics is better without them)""" + missing + """ Your best basis is also checked on independent data: it passes if no
single change (dropping, adding or swapping one operator) improves it, a simple greedy search does not beat it, and
no better basis is known there.
You keep proposing until it passes, and you are told why it failed. A failed check means a better basis exists: proposing
a basis you have already tried again brings nothing new and costs reward."""


MAX_SHOWN_LINES = 4  # lines of a reply quoted to the model; a reply that runs on can have dozens


def _shown(lines: tuple[str, ...]) -> str:
    shown = [f"- {line[:120]}" for line in lines[:MAX_SHOWN_LINES]]
    if len(lines) > MAX_SHOWN_LINES:
        shown.append(f"- and {len(lines) - MAX_SHOWN_LINES} more")
    return "\n".join(shown)


@dataclass
class Attempt:
    turn: int
    reply: str
    score: Score
    report: FlowReport
    found: bool = True  # whether the reply has a <basis> block
    invalid: tuple[str, ...] = ()  # its unusable lines, with the reason
    dropped: tuple[str, ...] = ()  # its lines that repeat an observable of the basis, with the reason
    repeated: float = 0.0  # the penalty for a basis already tried in the session, 0 for a new one
    invalid_cost: float = 0.0  # the penalty for its unusable lines
    duplicate_cost: float = 0.0  # the penalty for its repeated operators

    @property
    def total(self) -> float:
        return self.score.total

    def line(self) -> str:
        if not self.found:
            return f"no <basis> block: reward {self.total:+.3f}"
        text = (f"[{', '.join(('u', *self.report.texts))}]: error {self.score.error:.4f}, "
                f"cost x{self.score.relative_cost:.2f}, reward {self.total:+.3f}")
        if self.invalid or self.dropped or self.repeated:
            why = " and ".join(w for w, on in (("its unusable lines", self.invalid), ("its repeated operators", self.dropped),
                                                ("repeating a tried basis", self.repeated)) if on)
            text += f" ({reward(self.score):+.3f} for the basis, {self.score.format:+.3f} for {why})"
        return text

    def problems(self) -> str | None:
        """What was wrong with the format of the reply, in the words the model is told; None if nothing was."""
        if not self.found:
            return (f"Your latest reply has no <basis> block, which cost {self.score.format:+.3f} of reward. End every "
                    f"reply with the basis between <basis> and </basis>, one operator per line.")
        parts = []
        if self.invalid:
            lines = "1 unusable line" if len(self.invalid) == 1 else f"{len(self.invalid)} unusable lines"
            parts.append(f"Your latest reply has {lines}, ignored, which cost {-self.invalid_cost:+.3f} of reward:\n"
                         + _shown(self.invalid) + "\nWrite one operator per line, each one cheap enough to evaluate.")
        if self.dropped:
            parts.append(f"Your latest reply repeats observables already in its basis, dropped, which cost "
                         f"{-self.duplicate_cost:+.3f} of reward:\n" + _shown(self.dropped))
        if self.repeated:
            parts.append(f"Your latest reply repeats a basis you have already tried, which cost {-self.repeated:+.3f} of "
                         f"reward. Its result is already above: propose a basis you have not tried.")
        return "\n\n".join(parts) or None


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
                            error=a.score.error, 
                            cost=a.score.relative_cost)
                        for a in self.chosen
                        ],
                    best=list(self.best.score.names), 
                    best_reward=self.best.total, 
                    first_reward=self.chosen[0].total,
                    passed=self.passed, 
                    n_turns=len(self.chosen))


def session_messages(t: Task, history: list[Attempt], hints: Hints = "stats", system_role: bool = True, max_history: int = 6, check: Verification | None = None, listed: bool = True) -> list[dict]:
    """The prompt of a turn; check: the last check of the best basis, said when it failed; listed: whether the feedback
    lists missing monomials (LoopConfig.candidates > 0), which the system prompt then announces."""
    parts = [diagnostics(environment(t), hints)]
    if history:
        tried = {}
        for a in sorted(history, key=lambda a: -a.total):
            tried.setdefault(a.score.names, a)  # the same basis written twice is listed once
        parts.append("Bases you have tried, best first:\n" + "\n".join(
            f"- {a.line()}" for a in list(tried.values())[:max_history]))
        best, last = max(history, key=lambda a: a.total), history[-1]
        if last.problems():
            parts.append(last.problems())
        parts.append("RG flow of your best basis:\n" + best.report.text())
        if last.score.names != best.score.names:
            parts.append("RG flow of your latest basis:\n" + last.report.text())
        if check is not None and not check.passed:
            parts.append("Check of your best basis on independent data: failed, " + "; ".join(check.failures) + ".")
        parts.append("Propose a better basis: more accurate at the coarsest scale, or as accurate and cheaper.")
    else:
        parts.append("Propose the basis.")
    user = "\n\n".join(parts)
    system = system_loop(listed)
    if system_role:
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return [{"role": "user", "content": system + "\n\n" + user}]


@dataclass
class LoopConfig:
    rounds: int = 12  # each round runs a session on every training task
    max_turns: int = 4  # proposals per session at most, each after the feedback on the previous one; a session
    # ends earlier when its best basis passes the check
    group: int = 8  # replies sampled per turn, the GRPO group
    update_every: int = 2  # rounds between post-training updates
    eval_every: int = 1  # rounds between the greedy evaluations, at an update; the last round is always evaluated
    tasks_per_round: int | None = None  # training tasks of a round, taken in turn; None for all of them
    max_hours: float | None = None  # after this long, finish the round, evaluate, verify and stop; None for no limit
    hints: Hints = "stats"  # what the diagnostics show; the RG flow feedback lists the missing monomials anyway
    candidates: int = 5  # monomials outside the basis listed in the feedback, 0 for none
    reports: tuple[str, ...] = ()  # extra feedback lines, names of feedback.REPORTS, e.g. ("leak",)
    system_role: bool = True
    yardstick: bool = True  # greedy's basis on the reward ensemble, the yardstick of the advantage over greedy; False skips it
    search: SearchConfig = field(default_factory=SearchConfig)  # of the yardstick and of the check, and its space
    check: CheckConfig = field(default_factory=CheckConfig)  # what a basis must pass to end a session
    check_seed: int = VERIFY_SEED  # the ensemble of the check, whose verdict the model is told
    final_seed: int = FINAL_SEED  # the ensemble of the final verification, never seen by the loop
    output_dir: str = "runs/loop"
    reward: RewardConfig = field(default_factory=RewardConfig)


def attempt(t: Task, reply: str, config: LoopConfig, turn: int = 0, scored: Score | None = None, repeated: float = 0.0) -> Attempt:
    """A reply scored by the reward, with the RG flow report of its basis: what the model is told about it. scored: the
    score of the reply, if already computed, with the repeat penalty `repeated` already in its format part."""
    scored = score(t, reply, config.reward) if scored is None else scored
    env = environment(t, REWARD_SEED)
    basis = read_reply(reply, config.reward, env)
    report = flow_report(env, basis.ops, config.reward, config.candidates, config.reports)
    return Attempt(turn, reply, scored, report, basis.found, tuple(basis.invalid), tuple(basis.dropped), repeated,
                   config.reward.invalid_penalty * len(basis.invalid), config.reward.duplicate_penalty * len(basis.dropped))


def repeat_penalties(replies: list[str], scores: list[Score], tried: list[Attempt], config: RewardConfig, env: KernelEnvironment) -> list[float]:
    """The penalty of each reply for proposing a basis already tried in the session. A session goes on only after the
    check failed, so a repeat brings nothing new; without the penalty it scores what it scored before and outranks every
    honest try at a better basis, which teaches the model to repeat itself after a failure."""
    seen = {a.score.names for a in tried if a.found}
    return [config.repeat_penalty if s.names in seen and read_reply(r, config, env).found else 0.0
            for r, s in zip(replies, scores)]


def _check(t: Task, reply: str, config: LoopConfig, seed: int, **options) -> Verification:
    """verify.verify on the ensemble of `seed`, with the settings of the loop."""
    return verify(t, reply, config.reward, seed, search=config.search, check=config.check, **options)


def run_session(policy: Proposer, t: Task, config: LoopConfig, greedy: bool = False, round: int = 0, greedy_baseline: SearchResult | None = None) -> Session:
    """Proposals on one task until the best basis passes the check, at most config.max_turns of them. greedy_baseline: the check's baseline for the task, computed once by the loop."""
    session, checked = Session(t.name), {}
    for turn in range(config.max_turns):
        prompt = session_messages(t, session.chosen, config.hints, config.system_role, check=session.checks[-1] if session.checks else None,
                                  listed=config.candidates > 0)
        completions = policy.generate(prompt, 1 if greedy else config.group, greedy=greedy)
        scores = [score(t, c.text, config.reward) for c in completions]
        penalties = repeat_penalties([c.text for c in completions], scores, session.chosen, config.reward,
                                     environment(t, REWARD_SEED))
        scores = [replace(s, format=s.format - p) if p else s for s, p in zip(scores, penalties)]
        session.groups.append(Group(t.name, prompt, completions, [s.total for s in scores], round, turn))
        j = max(range(len(scores)), key=lambda i: scores[i].total)
        session.chosen.append(attempt(t, completions[j].text, config, turn, scores[j], penalties[j]))
        reply = session.best.reply
        if reply not in checked:  # the best basis may not change from one turn to the next
            checked[reply] = _check(t, reply, config, config.check_seed, measure=False, greedy_baseline=greedy_baseline)
        session.checks.append(checked[reply])
        if session.passed:
            break
    return session


def greedy_yardstick(t: Task, config: LoopConfig) -> Score:
    """The yardstick of the advantage over greedy: the basis greedy forward selection finds on the reward ensemble."""
    return greedy(environment(t, REWARD_SEED), config.reward, config.search).best


class DiscoveryLoop:
    def __init__(self, policy: Proposer, config: LoopConfig | None = None, train: list[Task] | None = None, unseen: list[Task] | None = None, log=print):
        self.policy, self.config, self.log = policy, LoopConfig() if config is None else config, log
        self.train = split("train") if train is None else train
        self.unseen = split("unseen") if unseen is None else unseen
        self.tasks = self.train + self.unseen
        self.out = Path(self.config.output_dir)
        if self.config.final_seed in (REWARD_SEED, self.config.check_seed):
            raise ValueError(f"final_seed {self.config.final_seed} must differ from the reward seed {REWARD_SEED} and from check_seed {self.config.check_seed}: the final verification needs data the loop has not seen")
        self.baselines: dict[str, SearchResult] = {}  # the check's greedy baseline per task, computed once
        self.history: list[dict] = []  # one entry per round and per evaluation
        self.yardsticks: dict[str, Score] = {}
        self.final: list[Session] = []
        self.rotation = random.Random(0).sample(self.train, len(self.train))  # the order the rounds take the tasks in

    def advantage(self, task: str, a: Attempt) -> float | None:
        """The advantage over greedy of an attempt: its reward minus the reward of greedy's basis on the reward ensemble.
        > 0: the model beats greedy, 0: it ties, < 0: greedy is better. None without a yardstick."""
        yardstick = self.yardsticks.get(task)
        return None if yardstick is None else a.total - reward(yardstick)

    def baseline(self, t: Task) -> SearchResult:
        if t.name not in self.baselines:
            cfg = self.config
            self.baselines[t.name] = baseline(t, cfg.reward, cfg.check_seed, cfg.search)
        return self.baselines[t.name]

    def evaluate(self, round: int) -> list[Session]:
        """An evaluation session on every task with the current weights: the most likely reply, no sampling."""
        sessions = []
        for t in self.tasks:
            s = run_session(self.policy, t, self.config, greedy=True, round=round, greedy_baseline=self.baseline(t))
            sessions.append(s)
            entry = self._record("eval", round, t, s)
            n_turns = len(s.chosen)
            outcome = (f"passed in {n_turns} turns" if s.passed else f"not passed in {n_turns} turns: " + "; ".join(s.checks[-1].failures))
            lines = [f"  [eval {t.split:5s}] {t.name:16s} {outcome}"]
            for label, a, adv in (("first", s.chosen[0], entry["first_advantage"]), ("best ", s.best, entry["best_advantage"])):
                lines.append(f"  {'':30s} {label} {a.line()}" + ("" if adv is None else f" advantage over greedy {adv:+.3f}"))
            self.log("\n".join(lines))
        return sessions

    def run(self, resume: bool = False) -> list[Verification]:
        """Sample, update and evaluate for config.rounds rounds, then verify on the final ensemble. resume: go on from
        the last update a run in config.output_dir saved (see resume_point), with its history; the policy must already
        hold that update's checkpoint, e.g. HFPolicy.from_pretrained(model, adapter=resume_point(output_dir)[1])."""
        cfg, start = self.config, time.time()
        first = 1
        if resume:
            point = resume_point(self.out)
            if point is None:
                raise ValueError(f"nothing to resume in {self.out}: no update in its history.json")
            first = point[0] + 1
            if first > cfg.rounds:
                raise ValueError(f"the run in {self.out} already did {point[0]} rounds: raise config.rounds to go on")
            self.history = [h for h in json.loads((self.out / "history.json").read_text()) if h["round"] < first]
        self.out.mkdir(parents=True, exist_ok=True)
        # the loop's settings, and the tasks it ran on with the size their data were made at
        tasks = {t.name: {"split": t.split, "size": dataclasses.asdict(t.size)} for t in self.tasks}
        (self.out / "loop_config.json").write_text(json.dumps(dataclasses.asdict(cfg) | {"tasks": tasks}, indent=2))
        if cfg.yardstick and not self.yardsticks:  # yardsticks set beforehand are kept
            self._find_yardsticks()
        sessions = None  # the last evaluation's sessions
        if resume:
            self.log(f"[resumed after round {first - 1} from {point[1]}]")
        else:
            self.log("[eval] before post-training")
            sessions = self.evaluate(0)
        buffer: list[Group] = []
        for r in range(first, cfg.rounds + 1):
            buffer += self._sample(r)
            out_of_time = cfg.max_hours is not None and time.time() - start > cfg.max_hours * 3600
            last = r == cfg.rounds or out_of_time
            if r % cfg.update_every == 0 or last:
                self._update(r, buffer)
                buffer = []  # the next groups come from the updated weights
                if r % cfg.eval_every == 0 or last:
                    sessions = self.evaluate(r)
                self._save(sessions)  # a crash later on loses at most the round after this one
            if out_of_time:
                self.log(f"[stopped after round {r}: over {cfg.max_hours:g} hours]")
                break
        self.final = sessions
        verifications = []
        for t, s in zip(self.tasks, sessions):
            # its own greedy baseline, on that seed; rivals from the training and from every check of the run
            v = _check(t, s.best.reply, cfg, cfg.final_seed, measure=True, rival_seeds=(REWARD_SEED, cfg.check_seed))
            release(t, cfg.final_seed)  # needed for this task alone: a third of the environments
            verifications.append(v)
            self.log(f"[verify {t.split}, final ensemble] " + ("" if s.passed else "(the session did not pass the check) ") + v.report())
        self._save(sessions, verifications)
        return verifications

    def _find_yardsticks(self) -> None:
        t0 = time.time()
        for t in self.tasks:
            self.yardsticks[t.name] = greedy_yardstick(t, self.config)
        found = ", ".join(f"{name} {list(s.names)}" for name, s in self.yardsticks.items())
        self.log(f"[yardsticks] greedy forward selection ({time.time() - t0:.0f}s): " + found)

    def round_tasks(self, r: int) -> list[Task]:
        """The training tasks of round r: all of them, or the next tasks_per_round of them in turn."""
        n = len(self.rotation)
        k = min(self.config.tasks_per_round or n, n)
        return [self.rotation[((r - 1) * k + i) % n] for i in range(k)]

    def _sample(self, r: int) -> list[Group]:
        """A sampled session on the training tasks of the round; returns their groups."""
        t0 = time.time()
        groups, entries = [], []
        for t in self.round_tasks(r):
            s = run_session(self.policy, t, self.config, round=r, greedy_baseline=self.baseline(t))
            groups += s.groups
            rewards = [x for g in s.groups for x in g.rewards]
            entries.append(self._record("sample", r, t, s, mean_reward=sum(rewards) / len(rewards),
                                        distinct=[g.distinct for g in s.groups]))
        mean = sum(e["mean_reward"] for e in entries) / len(entries)
        best = ", ".join(f"{e['task']} {e['best_reward']:+.3f}" for e in entries)
        passed = sum(e["passed"] for e in entries)
        tied = sum(g.distinct == 1 for g in groups)
        self.log(f"[round {r}] mean reward of the samples {mean:+.3f}, best per task {best}, {passed}/{len(entries)} "
                 f"sessions passed the check, {tied}/{len(groups)} groups with one reward for all replies (no signal) "
                 f"({time.time() - t0:.0f}s)")
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
        """Add a session to the history, with the advantage over greedy of its first and best proposals."""
        entry = dict(kind=kind, round=round, split=t.split, **s.summary(), **extra,
                     first_advantage=self.advantage(s.task, s.chosen[0]), best_advantage=self.advantage(s.task, s.best))
        self.history.append(entry)
        return entry

    def _save(self, sessions: list[Session] | None, verifications: list[Verification] | None = None) -> None:
        (self.out / "history.json").write_text(json.dumps(self.history, indent=1))
        if sessions is None:  # resumed, and not evaluated since: the last transcripts on disk stay
            return
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


def resume_point(output_dir: str | Path) -> tuple[int, Path] | None:
    """Where a run in output_dir can go on from: the round of the last update its history.json records, and that
    update's checkpoint; None if it records no update. A checkpoint the history does not know of (a crash between the
    two) is ignored. The optimizer's state is not saved: it starts afresh, as at round 1."""
    path = Path(output_dir) / "history.json"
    if not path.exists():
        return None
    rounds = [h["round"] for h in json.loads(path.read_text()) if h["kind"] == "update"]
    if not rounds:
        return None
    checkpoint = Path(output_dir) / f"checkpoint-{max(rounds)}"
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"{path} records the update of round {max(rounds)}, but {checkpoint} is missing")
    return max(rounds), checkpoint


def final_bases(output_dir: str | Path) -> dict[str, list[SymbolicOperator]]:
    """The basis the post-trained model ended with on every task of a finished loop, from its verification.json: {task: operators besides u}."""
    entries = json.loads((Path(output_dir) / "verification.json").read_text())
    missing = [e["task"] for e in entries if "operators" not in e]
    if missing:
        raise ValueError(f"{output_dir}/verification.json has no operators for {missing}: rerun the loop, or read the bases from last_eval_sessions.json")
    return {e["task"]: [SymbolicOperator(text) for text in e["operators"]] for e in entries}


def learning_curves(history: list[dict]) -> dict[str, dict[str, list[float]]]:
    """The evaluations of a loop (DiscoveryLoop.history) by round: for each split, the mean over its tasks of the
    advantage over greedy of the first proposal and of the best one, the reward of the best one, the fraction of sessions that passed
    the check and their number of turns; and the weight change after each update. {"train": {"round": [...],
    "first_advantage": [...], "best_advantage": [...], "best_reward": [...], "passed": [...], "n_turns": [...]},
    "unseen": {...}, "updates": {"round": [...], "weight_change": [...]}}"""
    curves: dict[str, dict[str, list[float]]] = {}
    evals = [h for h in history if h["kind"] == "eval"]
    for part in ("train", "unseen"):
        rows = [h for h in evals if h["split"] == part]
        rounds = sorted({h["round"] for h in rows})
        curves[part] = {"round": rounds}
        for key in ("first_advantage", "best_advantage", "best_reward", "passed", "n_turns"):
            values = [[float(h[key]) for h in rows if h["round"] == r and h.get(key) is not None] for r in rounds]
            curves[part][key] = [sum(v) / len(v) if v else float("nan") for v in values]
    updates = [h for h in history if h["kind"] == "update"]
    curves["updates"] = {
        "round": [h["round"] for h in updates],
        "weight_change": [h.get("weight_change", float("nan")) for h in updates]
        }
    return curves
