from dataclasses import dataclass, replace
from typing import Literal

from core.data.polynomial_pde import PolynomialPDE
from core.evaluation.environment import Levels
from core.evaluation.kernels import KernelEnvironment

# three independent ensembles (an environment of seed s simulates its fit trajectories from s, its scoring
# ones from s + 1): the reward and the RG flow feedback; the check that ends a session of the discovery loop, whose
# verdict the model is told; and the final verification, which the model never hears about
REWARD_SEED = 0
VERIFY_SEED = 100
FINAL_SEED = 200


@dataclass(frozen=True)
class EnvSize:
    n: int  # grid points of the simulation
    n_time: int  # snapshots per trajectory
    fit: tuple[int, ...]  # levels the kernels are fitted at; level n has block size 2^n
    predict: int  # the prediction scale: the level the kernels are extrapolated to and the effective dynamics predicts at
    n_fit: int  # trajectories the kernels are fitted on
    n_score: int  # scoring trajectories: the prediction error of the effective dynamics is measured on them, never fitted
    n_lags: int = 4  # memory of the kernels


# a laptop demo: ~6 s to build an environment, ~80 ms to score a basis
DEMO = EnvSize(n=128, n_time=21, fit=(1, 2, 3), predict=4, n_fit=100, n_score=50)
# the size of the original Burgers search (a long run)
FULL = EnvSize(n=256, n_time=21, fit=(1, 2, 3, 4), predict=5, n_fit=400, n_score=200)


@dataclass(frozen=True)
class Task:
    name: str
    rhs: str  # du/dt in the DSL of core.operators.symbolic: the ground truth, never shown to the model
    split: Literal["train", "unseen"]  # unseen: the language model is never trained on it, only evaluated
    size: EnvSize = DEMO  # how its data are made and scored: grid, snapshots, coarse levels, trajectories


TASKS = (
    # ---- training tasks: the language model is post-trained on them ----
    # Burgers, flux u^2/2: best with u**2. Several strengths and viscosities, so that a model cannot do well by
    # memorizing an answer per prompt
    Task("burgers", "-0.5*D(u**2) + 0.08*D(u,2)", "train"),
    Task("burgers_weak", "-0.25*D(u**2) + 0.08*D(u,2)", "train"),
    Task("burgers_viscous", "-0.5*D(u**2) + 0.16*D(u,2)", "train"),
    # modified Burgers, flux u^3: best with u**3
    Task("modified_burgers", "-D(u**3)/3 + 0.08*D(u,2)", "train"),
    Task("modified_strong", "-D(u**3) + 0.08*D(u,2)", "train"),
    Task("modified_thin", "-D(u**3)/3 + 0.04*D(u,2)", "train"),
    # both fluxes: best with two operators, u**2 and u**3, on all three ensembles, while the first-turn list puts only
    # one of them first, so they reward revising a basis from the feedback. On the first two the list puts u**2 first;
    # on the last three u**3, and the feedback on [u, u**3] puts u**2 first
    Task("burgers_cubic", "-0.5*D(u**2) - 0.3*D(u**3) + 0.08*D(u,2)", "train"),
    Task("burgers_cubic_viscous", "-0.5*D(u**2) - 0.3*D(u**3) + 0.16*D(u,2)", "train"),
    Task("cubic_dominant", "-0.2*D(u**2) - 0.8*D(u**3) + 0.08*D(u,2)", "train"),
    Task("cubic_dominant_viscous", "-0.2*D(u**2) - 0.6*D(u**3) + 0.16*D(u,2)", "train"),
    Task("cubic_strong", "-0.3*D(u**2) - 1.0*D(u**3) + 0.08*D(u,2)", "train"),
    # KPZ, nonlinearity u_x^2: mostly best with u alone
    Task("kpz", "-0.5*D(u)**2 + 0.08*D(u,2)", "train"),
    Task("kpz_strong", "-D(u)**2 + 0.08*D(u,2)", "train"),
    Task("kpz_burgers", "-0.2*D(u**2) - 0.5*D(u)**2 + 0.08*D(u,2)", "train"),
    # u alone is not best here, unlike on the other KPZ tasks: the feedback on [u] lists D(u)**2 and u*D(u)**2 first,
    # and the best basis has both
    Task("kpz_viscous", "-D(u)**2 + 0.16*D(u,2)", "train"),
    # other nonlinearities: a source term, and Burgers with dispersion
    Task("quadratic_source", "-0.3*u**2 + 0.08*D(u,2)", "train"),
    Task("kdv_strong", "-0.5*D(u**2) - 0.06*D(u,3) + 0.05*D(u,2)", "train"),
    # ---- unseen tasks: never trained on, only evaluated. In the same order of families; on some the right choice is
    # worth more than the noise between ensembles, on others it is u alone ----
    Task("burgers_strong", "-D(u**2) + 0.08*D(u,2)", "unseen"),
    Task("modified_mid", "-0.5*D(u**3) + 0.08*D(u,2)", "unseen"),
    Task("cubic_quadratic", "-0.2*D(u**2) - 0.5*D(u**3) + 0.08*D(u,2)", "unseen"),
    Task("kpz_weak", "-0.25*D(u)**2 + 0.08*D(u,2)", "unseen"),
    Task("burgers_kpz", "-0.5*D(u**2) - 0.5*D(u)**2 + 0.08*D(u,2)", "unseen"),
    Task("kdv_burgers", "-0.5*D(u**2) - 0.03*D(u,3) + 0.05*D(u,2)", "unseen"),
    Task("kuramoto_sivashinsky", "-0.5*D(u**2) - 0.05*D(u,2) - 0.01*D(u,4)", "unseen"),
)


def task(name: str) -> Task:
    return next(t for t in TASKS if t.name == name)


def split(name: Literal["train", "unseen"]) -> list[Task]:
    return [t for t in TASKS if t.split == name]


def as_task(t: "Task | str") -> Task:
    """A task, or the task of the bank with that name."""
    return t if isinstance(t, Task) else task(t)


def sized(tasks: list[Task], size: EnvSize) -> list[Task]:
    """The same tasks with their data made at another size, e.g. sized(split("train"), FULL) for a long run."""
    return [replace(t, size=size) for t in tasks]


_ENVIRONMENTS: dict[tuple[Task, int], KernelEnvironment] = {}


def environment(task: "Task | str", seed: int = REWARD_SEED) -> KernelEnvironment:
    """The kernel environment of a task (a Task, or the name of one of the bank) on the ensemble of a seed, at the
    task's size, without the per-operator cost of its score (reward.py prices operators by their simulation cost
    instead); cached."""
    key = (as_task(task), seed)  # one cache entry, however the arguments are spelled
    if key not in _ENVIRONMENTS:
        _ENVIRONMENTS[key] = _environment(*key)
    return _ENVIRONMENTS[key]


def release(task: "Task | str", seed: int = REWARD_SEED) -> None:
    """Drop an environment from the cache, for one that is no longer needed: ~90 MB each at the demo size."""
    _ENVIRONMENTS.pop((as_task(task), seed), None)


def _environment(task: Task, seed: int) -> KernelEnvironment:
    size = task.size
    env = KernelEnvironment(
        PolynomialPDE(task.rhs, n=size.n, n_time=size.n_time),
        Levels(fit=size.fit, predict=size.predict, n_extrapolation_levels=len(size.fit)),
        n_fit=size.n_fit,
        n_score=size.n_score,
        seed=seed,
        cost=0.0,
        n_lags=size.n_lags,
    )
    # a run keeps one environment per task and ensemble, ~72 of them: the default noise cache made them ~170 MB each.
    # Two bases of noise are enough for the searches here, as fast and with the same results.
    env.noise_cache_size = 8
    return env
