from dataclasses import dataclass, replace

import numpy as np

from core.inference.mori_kernels import MemoryKernels, observables
from core.operators.symbolic import SymbolicOperator
from core.evaluation.environment import Environment, Levels, extrapolate


@dataclass
class Leak:
    """How far the span of the observables g = (u, O_1, ..., O_p) is from invariant under the coarse-grained dynamics.
    An exactly invariant span has rows = 0 (every step of g is a combination of g) and memory = 0 (Markov kernels)."""
    names: tuple[str, ...]  # the operators besides u
    level: int
    scoring: bool  # True: extrapolated kernels on the scoring trajectories; False: kernels fitted at `level`, on the fit trajectories
    rows: np.ndarray  # (M,) per observable, |W_i| / |g_i(t+1) - g_i(t)|: 0 predicted exactly, 1 no better than "no change"
    memory: np.ndarray  # (M,) per observable, the share of the prediction carried by the lags l >= 1

    @property
    def u(self) -> float:
        return float(self.rows[0])


def _leak(kernels: np.ndarray, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """rows and memory of Leak from kernels (n_lags, K, M, M) and trajectories g (batch, time, K, M) at one level."""
    h = kernels.shape[0]
    windows = np.lib.stride_tricks.sliding_window_view(g[:, :-1], h, axis=1)  # windows[b, s, k, j, w] = g_j(s + w)
    prediction = np.einsum("lkij,bskjl->bski", kernels[::-1], windows, optimize=True)
    markov = np.einsum("kij,bskj->bski", kernels[0], g[:, h - 1:-1])
    norm = lambda x: np.linalg.norm(x.reshape(-1, x.shape[-1]), axis=0)  # per observable, over batch, time and modes
    with np.errstate(all="ignore"):
        return norm(g[:, h:] - prediction) / norm(g[:, h:] - g[:, h - 1:-1]), norm(prediction - markov) / norm(prediction)


def _complex(z: complex) -> str:
    return f"{z.real:+.4g}{z.imag:+.4g}j"


@dataclass
class KernelEvaluation:
    names: tuple[str, ...]  # the operators besides u
    score: float
    error: float  # rollout error at the prediction scale with extrapolated kernels
    direct_error: float  # same, with kernels fitted at the prediction scale directly (what extrapolation could reach)
    kernels: np.ndarray  # (n_fit_levels, n_lags, n_k, M, M) at the wavevectors of the prediction scale
    extrapolated: np.ndarray  # (n_lags, n_k, M, M)
    direct: np.ndarray  # (n_lags, n_k, M, M) fitted at the prediction scale
    wavevectors: np.ndarray  # (n_k, n_space) the wavevectors of the prediction scale, in the order of the kernels
    residual: np.ndarray  # (n_fit_levels,) one-step |W_u| / |u_k(t+1) - u_k(t)| on the fit trajectories
    levels: Levels

    def describe(self) -> str:
        # the kernels are reported at the wavevector closest to k = (1, 0, ..., 0)
        unit = np.eye(self.wavevectors.shape[1])[0]
        q = int(np.argmin(np.linalg.norm(self.wavevectors - unit, axis=1)))
        k = self.wavevectors[q]
        k_label = f"{k[0]:g}" if len(k) == 1 else "(" + ", ".join(f"{c:g}" for c in k) + ")"
        labels = ", ".join(("u",) + self.names)
        lines = [f"observables [{labels}]: score {self.score:+.3f}, rollout error at the prediction scale {self.error:.4f} "
                 f"(kernels fitted at that level directly: {self.direct_error:.4f})"]
        for i, level in enumerate(self.levels.fit):
            ks = ", ".join(_complex(z) for z in self.kernels[i, 0, q, 0])
            lines.append(f"  level {level} (block size {2**level:>2}): Markov kernel into u_k at k={k_label} [{ks}], "
                         f"unexplained u step {self.residual[i]:.4f}")
        ks = ", ".join(_complex(z) for z in self.extrapolated[0, q, 0])
        kd = ", ".join(_complex(z) for z in self.direct[0, q, 0])
        predict = self.levels.predict
        lines.append(f"  level {predict} (block size {2**predict}): extrapolated [{ks}], fitted directly [{kd}]")
        return "\n".join(lines)


class KernelEnvironment(Environment[KernelEvaluation]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # every wavevector of the grid of the prediction scale, (n_k, n_space), in the order of the kernel modes (meshgrid "ij" of the per-axis fftfreq wavenumbers, as in MoriKernels)
        axes = self.u[self.levels.predict].wavenumbers()
        self.k_predict = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, len(axes))

    def prepare(self, ops: list[SymbolicOperator]) -> list[SymbolicOperator]:
        return self.observable_ops(ops)

    def at_predict_wavevectors(self, kernels: MemoryKernels) -> np.ndarray:
        k = kernels.wavevectors  # (K, n_space)
        idx = np.array([np.argmin(np.linalg.norm(k - q, axis=1)) for q in self.k_predict])
        assert np.allclose(k[idx], self.k_predict)
        return kernels.kernels[:, idx]

    def rollout_error(self, kernels: MemoryKernels, level: int) -> float:
        """The rollout error of the kernels at the prediction scale: the relative error of the predicted u_k(t) vs the true u_k(t) at the wavevectors of the prediction scale, averaged over time and wavevectors. The rollout is done by iteratively predicting the next time step from the previous h steps, where h = kernels.n_lags."""
        c = self.u_score[level]
        assert c.time is not None, "the scoring trajectories have a time axis"
        true, h = c.values, kernels.n_lags
        u, errors = true[:, :h], []
        with np.errstate(all="ignore"):
            for t in range(h, self.dynamics.n_time):
                window = c.replace(values=u[:, -h:], time=c.time[t - h:t])
                uk = kernels.predict(observables(window, kernels.basis))[..., 0]  # (batch, K), K flattened modes
                uk = uk.reshape(len(uk), *c.spatial_shape)  # back onto the grid, "ij" order as the modes
                new = c.replace(values=uk, time=None, domain="fourier").to_real().values
                u = np.concatenate([u, new[:, None]], axis=1)
                err = np.linalg.norm(new - true[:, t]) / np.linalg.norm(true[:, t])
                errors.append(err if np.isfinite(err) else np.inf)
                if errors[-1] > self.max_error:
                    return self.max_error
        return float(min(np.mean(errors), self.max_error))

    def flow(self, ops: list[SymbolicOperator]) -> np.ndarray:
        """The kernels at the wavevectors of the prediction scale at every fit level -> (n_fit_levels, n_lags, n_k, M, M)."""
        return np.stack([self.at_predict_wavevectors(self.kernels(ops, n)) for n in self.levels.fit])

    def extrapolated_kernels(self, ops: list[SymbolicOperator]) -> MemoryKernels:
        """The kernels extrapolated along the flow to the prediction scale."""
        ops = self.prepare(ops)
        levels = self.levels
        lv = np.array(levels.fit[-levels.n_extrapolation_levels:], dtype=float)
        flow = self.flow(ops)[-levels.n_extrapolation_levels:]
        return replace(self.kernels(ops, levels.predict), kernels=extrapolate(flow, lv, levels.predict))

    def leak(self, ops: list[SymbolicOperator], level: int | None = None) -> Leak:
        """How far the span of (u, ops) is from invariant (see Leak). level None: at the prediction scale, with the
        kernels extrapolated along the flow, on the scoring trajectories, as the rollout error is scored; a fit level:
        with the kernels fitted there, on the fit trajectories (in-sample, so it favours larger bases)."""
        ops = self.prepare(ops)
        if level is None:
            kernels = self.extrapolated_kernels(ops)
            rows, memory = _leak(kernels.kernels, observables(self.u_score[self.levels.predict], kernels.basis))
            return Leak(tuple(op.name for op in ops), self.levels.predict, True, rows, memory)
        rows, memory = _leak(self.kernels(ops, level).kernels, self.basis_modes(ops, level))
        return Leak(tuple(op.name for op in ops), level, False, rows, memory)

    def _evaluate(self, ops: list[SymbolicOperator]) -> KernelEvaluation:
        levels = self.levels
        kernels = self.flow(ops)
        residual = np.array([self.one_step_residual(ops, n) for n in levels.fit])
        direct = self.kernels(ops, levels.predict)
        extrapolated = self.extrapolated_kernels(ops)
        error = self.rollout_error(extrapolated, levels.predict)
        return KernelEvaluation(
            names=tuple(op.name for op in ops),
            score=-np.log(error) - self.cost * len(ops),
            error=error,
            direct_error=self.rollout_error(direct, levels.predict),
            kernels=kernels,
            extrapolated=extrapolated.kernels,
            direct=direct.kernels,
            wavevectors=self.k_predict,
            residual=residual,
            levels=levels,
        )
