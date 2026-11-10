"""
Reading finished discovery-loop runs (the directories under runs/) for the lessons notebook: what their evaluations
reached, how much the feedback turns added, how stable the training was, how much room greedy left, and the bars that
fixed answers, scored without a model, set on the same tasks.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.operators.symbolic import distinct_monomials
from posttrain.reward import RewardConfig, score
from posttrain.tasks import Task, environment

SPLITS = ("train", "unseen")


@dataclass
class Run:
    name: str
    config: dict  # loop_config.json
    history: list[dict]  # history.json: one entry per sampled session ("sample"), evaluation session ("eval") and update
    verification: list[dict]  # verification.json: the final basis of every task on the final ensemble

    @property
    def evals(self) -> list[dict]:
        return [h for h in self.history if h["kind"] == "eval"]

    @property
    def updates(self) -> list[dict]:
        return [h for h in self.history if h["kind"] == "update"]

    @property
    def last_round(self) -> int:
        return max(h["round"] for h in self.evals)

    def round_evals(self, round: int | None = None) -> list[dict]:
        """The evaluation sessions of an evaluation round, the last one by default."""
        r = self.last_round if round is None else round
        return [h for h in self.evals if h["round"] == r]

    def split_of(self, task: str) -> str:
        return next(h["split"] for h in self.evals if h["task"] == task)


def load_run(path: str | Path, history: str = "history.json") -> Run:
    path = Path(path)
    read = lambda name: json.loads((path / name).read_text())
    return Run(path.name, read("loop_config.json"), read(history), read("verification.json"))


def load_runs(root: str | Path = "runs", pattern: str = "20*") -> list[Run]:
    """Every archived run under root, oldest first (the directory names start with their date)."""
    return [load_run(p) for p in sorted(Path(root).glob(pattern)) if (p / "history.json").exists()]


def summary(run: Run, round: int | None = None) -> dict:
    """Per split, at an evaluation round (the last by default): the number of tasks, the mean advantage over greedy of the first and
    the best proposal, and the sessions whose check passed; and the final verification passes."""
    out = {"run": run.name, "rounds": run.last_round, "updates": len(run.updates)}
    evals = run.round_evals(round)
    for part in SPLITS:
        rows = [h for h in evals if h["split"] == part]
        verified = [v for v in run.verification if run.split_of(v["task"]) == part]
        out[part] = dict(tasks=len(rows), first_advantage=_mean(h["first_advantage"] for h in rows),
                         best_advantage=_mean(h["best_advantage"] for h in rows), passed=sum(h["passed"] for h in rows),
                         verified=sum(v["passed"] for v in verified))
    return out


def revision(run: Run, round: int | None = None) -> dict:
    """What the turns after the first added, per split, at an evaluation round: sessions that went past turn 1, their
    mean turn gain (best minus first advantage over greedy), the share of them whose best proposal beat the first, and how many
    turns repeated the basis of the turn before."""
    out = {}
    for part in SPLITS:
        rows = [h for h in run.round_evals(round) if h["split"] == part]
        revised = [h for h in rows if h["n_turns"] > 1]
        gains = [h["best_advantage"] - h["first_advantage"] for h in revised]
        repeats = sum(a["basis"] == b["basis"] for h in revised for a, b in zip(h["turns"], h["turns"][1:]))
        out[part] = dict(sessions=len(rows), revised=len(revised), turn_gain=_mean(gains),
                         improved=_mean(g > 1e-9 for g in gains), repeats=repeats,
                         later_turns=sum(len(h["turns"]) - 1 for h in revised))
    return out


def moves(run: Run, round: int | None = None) -> list[dict]:
    """The bases of every evaluation session of an evaluation round, turn by turn, with the advantage over greedy of its
    first and best proposals."""
    return [dict(task=h["task"], split=h["split"], turns=[t["basis"] for t in h["turns"]],
                 first_advantage=h["first_advantage"], best_advantage=h["best_advantage"], passed=h["passed"])
            for h in run.round_evals(round)]


def training(run: Run) -> dict:
    """How the updates went: those that trained on no reply with signal (a collapsed policy), sampled groups whose
    replies all had one reward (when the run recorded it), the largest KL and loss, and updates with a non-finite loss."""
    ups = run.updates
    groups = [d for h in run.history if h["kind"] == "sample" for d in h.get("distinct", [])]
    losses = [u["loss"] for u in ups if "loss" in u]  # an update with nothing to train on records no loss
    return dict(updates=len(ups), empty_updates=sum(u["completions"] == 0 for u in ups),
                groups=len(groups) or None, tied_groups=sum(d == 1 for d in groups) if groups else None,
                max_kl=max((u["kl"] for u in ups if "kl" in u), default=None),
                max_loss=max((x for x in losses if np.isfinite(x)), default=None),
                nan_updates=sum(not np.isfinite(x) for x in losses),
                weight_change=ups[-1]["weight_change"] if ups else None)


def headroom(run: Run) -> dict[str, float]:
    """Per task, how much better than greedy's basis the best basis known on the final ensemble is: 0 means greedy
    already found the best known basis, so a proposal can at most tie it."""
    return {v["task"]: v["best_reward"] - v["baseline_reward"] for v in run.verification}


def list_rule(t: Task) -> str:
    """The reply of the rule a model can learn from the first prompt's operator list (hints="list"): the top
    monomial if it is u**2 or u**3, else u alone."""
    env = environment(t)
    pool = distinct_monomials()
    relevance, _, _ = env.candidate_features([], pool, env.levels.fit[0])
    top = pool[int(np.argmax(relevance))].text
    return f"<basis>\n{top}\n</basis>" if top in ("u**2", "u**3") else "<basis></basis>"


def fixed_advantages(tasks: list[Task], reply, yardsticks: dict[str, float], config: RewardConfig = RewardConfig()
                  ) -> dict[str, float]:
    """The advantage over greedy of a policy that needs no model, task by task: reply is a fixed reply or a function of
    the task; yardsticks: the reward of greedy's basis per task (loop.greedy_yardstick), as the loop uses."""
    out = {}
    for t in tasks:
        text = reply(t) if callable(reply) else reply
        out[t.name] = score(t, text, config).total - yardsticks[t.name]
    return out


def _mean(values) -> float:
    values = [float(v) for v in values]
    return float(np.mean(values)) if values else float("nan")
