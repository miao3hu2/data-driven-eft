from dataclasses import dataclass
from functools import cache
from typing import Literal

from core.data.polynomial_pde import PolynomialPDE
from core.evaluation.environment import Levels
from core.evaluation.kernels import KernelEnvironment

# three independent ensembles (an environment of seed s simulates its training trajectories from s, its held-out
# ones from s + 1): the reward and the RG flow feedback; the check that ends a session of the discovery loop, whose
# verdict the model is told; and the final verification, which the model never hears about
REWARD_SEED = 0
VERIFY_SEED = 100
FINAL_SEED = 200


@dataclass(frozen=True)
class Task:
    name: str
    rhs: str  # du/dt in the DSL of core.operators.symbolic: the ground truth, never shown to the model
    split: Literal["train", "test"]


TASKS = (
    Task("burgers", "-0.5*D(u**2) + 0.08*D(u,2)", "train"),
    Task("modified_burgers", "-D(u**3)/3 + 0.08*D(u,2)", "train"),
    Task("kpz", "-0.5*D(u)**2 + 0.08*D(u,2)", "train"),
    Task("burgers_cubic", "-0.5*D(u**2) - 0.3*D(u**3) + 0.08*D(u,2)", "train"),
    Task("kdv_burgers", "-0.5*D(u**2) - 0.03*D(u,3) + 0.05*D(u,2)", "test"),
    Task("burgers_kpz", "-0.5*D(u**2) - 0.5*D(u)**2 + 0.08*D(u,2)", "test"),
)


def task(name: str) -> Task:
    return next(t for t in TASKS if t.name == name)


def split(name: Literal["train", "test"]) -> list[Task]:
    return [t for t in TASKS if t.split == name]


@dataclass(frozen=True)
class EnvSize:
    n: int  # grid points of the simulation
    n_time: int  # snapshots per trajectory
    fit: tuple[int, ...]  # levels the kernels are fitted at; level n has block size 2^n
    heldout: int  # the level the kernels are extrapolated to and scored at
    n_train: int  # trajectories the kernels are fitted on
    n_test: int  # held-out trajectories
    n_lags: int = 4  # memory of the kernels


# a laptop demo: ~6 s to build an environment, ~80 ms to score a basis
DEMO = EnvSize(n=128, n_time=21, fit=(1, 2, 3), heldout=4, n_train=100, n_test=50)
# the size of the original Burgers search (a long run)
FULL = EnvSize(n=256, n_time=21, fit=(1, 2, 3, 4), heldout=5, n_train=400, n_test=200)
SIZES = {"demo": DEMO, "full": FULL}


@cache
def environment(task: Task, size: EnvSize = DEMO, seed: int = REWARD_SEED) -> KernelEnvironment:
    """The kernel environment of a task, without the per-operator cost of its score (reward.py prices operators by their simulation cost instead); cached."""
    return KernelEnvironment(
        PolynomialPDE(task.rhs, n=size.n, n_time=size.n_time),
        Levels(fit=size.fit, heldout=size.heldout, n_extrapolation_levels=len(size.fit)),
        n_train=size.n_train,
        n_test=size.n_test,
        seed=seed,
        cost=0.0,
        n_lags=size.n_lags,
    )
