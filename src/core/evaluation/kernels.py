from dataclasses import dataclass, replace

import numpy as np

from core.inference.mori_kernels import MemoryKernels, observables
from core.operators.symbolic import SymbolicOperator
from core.evaluation.environment import Environment, Levels, extrapolate


def _complex(z: complex) -> str:
    return f"{z.real:+.4g}{z.imag:+.4g}j"


@dataclass
class KernelEvaluation:
    names: tuple[str, ...]  # the operators besides u
    score: float
    heldout_error: float  # rollout error at the held-out level with extrapolated kernels
    direct_error: float  # same, with kernels fitted at the held-out level directly (what extrapolation could reach)
    kernels: np.ndarray  # (n_fit_levels, n_lags, n_k, M, M) at the held-out wavevectors
    extrapolated: np.ndarray  # (n_lags, n_k, M, M)
    direct: np.ndarray  # (n_lags, n_k, M, M) fitted at the held-out level
    wavevectors: np.ndarray  # (n_k, n_space) the held-out wavevectors, in the order of the kernels
    residual: np.ndarray  # (n_fit_levels,) one-step |W_u| / |u_k(t+1) - u_k(t)| on the training set
    levels: Levels

    def describe(self) -> str:
        # the kernels are reported at the wavevector closest to k = (1, 0, ..., 0)
        unit = np.eye(self.wavevectors.shape[1])[0]
        q = int(np.argmin(np.linalg.norm(self.wavevectors - unit, axis=1)))
        k = self.wavevectors[q]
        k_label = f"{k[0]:g}" if len(k) == 1 else "(" + ", ".join(f"{c:g}" for c in k) + ")"
        labels = ", ".join(("u",) + self.names)
        lines = [f"observables [{labels}]: score {self.score:+.3f}, held-out rollout error {self.heldout_error:.4f} "
                 f"(kernels fitted at that level directly: {self.direct_error:.4f})"]
        for i, level in enumerate(self.levels.fit):
            ks = ", ".join(_complex(z) for z in self.kernels[i, 0, q, 0])
            lines.append(f"  level {level} (b={2**level:>2}): Markov kernel into u_k at k={k_label} [{ks}], "
                         f"unexplained u step {self.residual[i]:.4f}")
        ks = ", ".join(_complex(z) for z in self.extrapolated[0, q, 0])
        kd = ", ".join(_complex(z) for z in self.direct[0, q, 0])
        heldout = self.levels.heldout
        lines.append(f"  level {heldout} (b={2**heldout}): extrapolated [{ks}], fitted directly [{kd}]")
        return "\n".join(lines)


class KernelEnvironment(Environment[KernelEvaluation]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # every wavevector of the held-out grid, (n_k, n_space), in the order of the kernel modes (meshgrid "ij" of the per-axis fftfreq wavenumbers, as in MoriKernels)
        axes = self.u[self.levels.heldout].wavenumbers()
        self.k_heldout = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, len(axes))

    def prepare(self, ops: list[SymbolicOperator]) -> list[SymbolicOperator]:
        return self.observable_ops(ops)

    def at_heldout_wavevectors(self, kernels: MemoryKernels) -> np.ndarray:
        k = kernels.wavevectors  # (K, n_space)
        idx = np.array([np.argmin(np.linalg.norm(k - q, axis=1)) for q in self.k_heldout])
        assert np.allclose(k[idx], self.k_heldout)
        return kernels.kernels[:, idx]

    def rollout_error(self, kernels: MemoryKernels, level: int) -> float:
        """The rollout error of the kernels at the held-out level: the relative error of the predicted u_k(t) vs the true u_k(t) at the held-out wavevectors, averaged over time and wavevectors. The rollout is done by iteratively predicting the next time step from the previous h steps, where h = kernels.n_lags."""
        c = self.u_test[level]
        assert c.time is not None, "the held-out trajectories have a time axis"
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
        """The kernels at the held-out wavevectors at every fit level -> (n_fit_levels, n_lags, n_k, M, M)."""
        return np.stack([self.at_heldout_wavevectors(self.kernels(ops, n)) for n in self.levels.fit])

    def extrapolated_kernels(self, ops: list[SymbolicOperator]) -> MemoryKernels:
        """The kernels extrapolated along the flow to the held-out level."""
        ops = self.prepare(ops)
        levels = self.levels
        lv = np.array(levels.fit[-levels.n_extrapolation_levels:], dtype=float)
        flow = self.flow(ops)[-levels.n_extrapolation_levels:]
        return replace(self.kernels(ops, levels.heldout), kernels=extrapolate(flow, lv, levels.heldout))

    def _evaluate(self, ops: list[SymbolicOperator]) -> KernelEvaluation:
        levels = self.levels
        kernels = self.flow(ops)
        residual = np.array([self.one_step_residual(ops, n) for n in levels.fit])
        direct = self.kernels(ops, levels.heldout)
        extrapolated = self.extrapolated_kernels(ops)
        heldout = self.rollout_error(extrapolated, levels.heldout)
        return KernelEvaluation(
            names=tuple(op.name for op in ops),
            score=-np.log(heldout) - self.cost * len(ops),
            heldout_error=heldout,
            direct_error=self.rollout_error(direct, levels.heldout),
            kernels=kernels,
            extrapolated=extrapolated.kernels,
            direct=direct.kernels,
            wavevectors=self.k_heldout,
            residual=residual,
            levels=levels,
        )
