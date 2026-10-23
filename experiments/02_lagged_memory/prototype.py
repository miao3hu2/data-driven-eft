"""Prototype: a lagged memory kernel for the coarse dynamics of the toy models in notebooks/01.

Discrete-time, data-driven Mori-Zwanzig: mode by mode,

    u_k[t+1] - u_k[t] = sum_{s=0..S} sum_i K_{s,i}(k) O_i(u[t-s])_k + noise_k[t]

s = 0 is the Markov term, s >= 1 the memory: the past resolved states stand in for the unresolved modes. The
coefficients are constant in time and fitted by least squares over all batch members and time steps t >= S_MAX, one
regression per wavevector (homogeneous) or one for all modes at once (homogeneous=False, every output mode on every
input mode and lag), optionally with a ridge penalty. Fitting S = 0, 1, ..., S_MAX on the same steps isolates what the history adds; the Markov model
and the gamma(t) model of mori_zwanzig_data are scored on the same window for reference.

Run from the repository root: uv run python experiments/02_lagged_memory/prototype.py
"""

import sys
from pathlib import Path

import numpy as np
import scipy.linalg

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from core.coarse_graining.filter import BlockCutoffFilter
from core.data.base import Field
from core.operators.base import Identity, Operator, OperatorBasis
from core.rgflow.memory import mori_zwanzig_data

# ---- the toy models of notebooks/01_mori_zwanzig_toy_models.ipynb (same parameters and seeds) ----

L, N = 2 * np.pi, 128
dx = L / N
x = np.arange(N) * dx
k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
DT, N_TIME = 0.05, 16
TIMES = np.arange(N_TIME) * DT
NU = 0.08
N_TRAIN, N_TEST = 600, 300


def random_initial_states(rng, n_batch, k0=3.0, amplitude=1.2):
    kk = np.abs(k)
    spectrum = (kk > 0) * kk**2 / (1 + (kk / k0) ** 2) ** 2
    modes = np.sqrt(spectrum) * rng.normal(size=(n_batch, N)) * np.exp(2j * np.pi * rng.random((n_batch, N)))
    u = np.real(np.fft.ifft(modes, axis=-1))
    return amplitude * u / u.std()


def burgers(u0, h=1e-3):
    dealias = np.abs(k) < (2 / 3) * np.abs(k).max()

    def rhs(uk):
        u = np.real(np.fft.ifft(uk, axis=-1))
        return -0.5j * k * np.fft.fft(u * u, axis=-1) * dealias - NU * k**2 * uk

    uk, u = np.fft.fft(u0, axis=-1), [u0]
    for _ in range(N_TIME - 1):
        for _ in range(round(DT / h)):
            k1 = rhs(uk)
            k2 = rhs(uk + h / 2 * k1)
            k3 = rhs(uk + h / 2 * k2)
            k4 = rhs(uk + h * k3)
            uk = uk + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        u.append(np.real(np.fft.ifft(uk, axis=-1)))
    return np.stack(u, axis=1)


def nonuniform_advection(u0, a=1.0):
    F = np.fft.fft(np.eye(N), axis=0)
    d1 = np.real(np.linalg.solve(F, np.diag(1j * k) @ F))
    d2 = np.real(np.linalg.solve(F, np.diag(-(k**2)) @ F))
    step = scipy.linalg.expm(DT * (NU * d2 + a * np.diag(np.cos(x)) @ d1))
    u = [u0]
    for _ in range(N_TIME - 1):
        u.append(u[-1] @ step.T)
    return np.stack(u, axis=1)


def ensemble(model, seed, n_batch):
    u = model(random_initial_states(np.random.default_rng(seed), n_batch))
    return Field(values=u, scale=dx, batched=True, time=TIMES)


def coarse(field, level):
    return BlockCutoffFilter(block_size=2**level, odd_grid=True).transform(field.to_fourier()).to_real()


class Advection(Operator):
    """-(u^2 / 2)_x on the field's own grid, dealiased by zero-padding."""

    name = "-(u^2/2)_x"

    def evaluate(self, field):
        u = field.to_real()
        m = u.spatial_shape[0]
        kx = u.wavenumbers()[0]
        modes = np.rint(kx * u.scale[0] * m / (2 * np.pi)).astype(int)
        padded = np.zeros(u.values.shape[:-1] + (2 * m,), dtype=complex)
        padded[..., modes % (2 * m)] = np.fft.fft(u.values, axis=-1)
        fine = np.real(np.fft.ifft(padded, axis=-1)) * 2
        square = np.fft.fft(fine**2 / 2, axis=-1)[..., modes % (2 * m)] / 2
        out = u.replace(values=np.real(np.fft.ifft(-1j * kx * square, axis=-1)))
        return out.to_fourier() if field.domain == "fourier" else out


# ---- the lagged model ----


def features(basis, values, scale):
    """Fourier modes of the operators: values (batch, [time,] m) real -> (batch, [time,] m, n_ops) complex."""
    time = None if values.ndim == 2 else np.arange(values.shape[1])
    field = Field(values=values, scale=scale, batched=True, time=time)
    return np.fft.fft(basis.stack(field).values, axis=-2)


class LaggedModel:
    """u_k[t+1] - u_k[t] = sum_{s=0..lags} K_s O(u[t-s])_k, constant coefficients fitted on the steps t >= t0.

    ridge > 0 minimises |y - X c|^2 + ridge * |D c|^2 instead, with D the rms of each regressor, so ridge is relative.
    Without it, the fit can be unstable in a rollout although it predicts one step well (Burgers with [u, -(u^2/2)_x]
    at level 1: the linear part of the fitted recurrence has spectral radius up to 3.3 there).
    """

    def __init__(self, basis, lags, homogeneous=True, ridge=0.0):
        self.basis, self.lags, self.homogeneous, self.ridge = basis, lags, homogeneous, ridge

    def _design(self, F, t):
        """Regressors at steps t: (batch, len(t), m, lags + 1, n_ops) from the features F (batch, time, m, n_ops)."""
        return np.stack([F[:, t - s] for s in range(self.lags + 1)], axis=-2)

    def fit(self, c, t0):
        self.scale = c.scale
        u = c.values
        F = features(self.basis, u, c.scale)
        t = np.arange(t0, u.shape[1] - 1)
        X = self._design(F, t)  # (batch, nt, m, lags + 1, n_ops)
        y = np.fft.fft(u[:, t + 1] - u[:, t], axis=-1)  # (batch, nt, m)
        m = y.shape[-1]
        if self.homogeneous:
            X = X.reshape(-1, m, X.shape[-2] * X.shape[-1])
            y = y.reshape(-1, m)
            self.coeffs = np.stack([self._solve(X[:, q], y[:, q]) for q in range(m)])  # (m, p)
        else:
            X = X.reshape(-1, m * X.shape[-2] * X.shape[-1])
            self.coeffs = self._solve(X, y.reshape(-1, m))  # (m * p, m)
        return self

    def _solve(self, X, y):
        if self.ridge == 0:
            return np.linalg.lstsq(X, y, rcond=None)[0]
        # the ridge as extra rows sqrt(ridge * n) D c = 0; a regressor that is zero everywhere gets coefficient 0
        rms = np.sqrt(np.mean(np.abs(X) ** 2, axis=0))
        penalty = np.diag(np.sqrt(self.ridge * len(X)) * rms)
        zeros = np.zeros((len(penalty), *y.shape[1:]), dtype=y.dtype)
        return np.linalg.lstsq(np.concatenate([X, penalty]), np.concatenate([y, zeros]), rcond=None)[0]

    def drift(self, F_hist):
        """The step's modes from the features of the last lags + 1 states, newest last: (batch, m)."""
        X = np.stack([F_hist[:, -1 - s] for s in range(self.lags + 1)], axis=-2)  # (batch, m, lags + 1, n_ops)
        b, m = X.shape[:2]
        if self.homogeneous:
            return np.einsum("bmp,mp->bm", X.reshape(b, m, -1), self.coeffs)
        return X.reshape(b, -1) @ self.coeffs

    def one_step(self, c, t):
        """Predicted steps u[t+1] - u[t] at the steps t from the true states: (batch, len(t), m) real."""
        F = features(self.basis, c.values, c.scale)
        return np.stack([np.real(np.fft.ifft(self.drift(F[:, s - self.lags : s + 1]), axis=-1)) for s in t], axis=1)

    def rollout(self, c, t0, t1):
        """From the true states u[0..t0], step the model to u[t1]: the states (batch, t1 - t0 + 1, m)."""
        u = [c.values[:, s] for s in range(t0 + 1)]
        for _ in range(t0, t1):
            hist = np.stack(u[-self.lags - 1 :], axis=1)
            F = features(self.basis, hist, c.scale)
            u.append(u[-1] + np.real(np.fft.ifft(self.drift(F), axis=-1)))
        return np.stack(u[t0:], axis=1)


def correlation_check(c, basis, lags, t0, ridge=0.0):
    """The homogeneous least-squares fit equals the solution of the correlation (normal) equations, mode by mode.

    With <.> the mean over batch members and steps t >= t0, and phi[t] = (O_i(u[t-s])_k) for s = 0..lags,
        C_k = <phi[t]^* phi[t]^T>    two-time correlations of the features, at lags s and s'
        b_k = <phi[t]^* du_k[t]>     correlations of the step du[t] = u[t+1] - u[t] with the lagged features
        (C_k + ridge diag(C_k)) K_k = b_k.
    Under stationarity C_k would be block-Toeplitz (Yule-Walker); its entries are averages of C(t - s, t - s') here.
    Returns, over the modes, the largest relative difference between the coefficients and between the predicted steps
    of the two solutions, and the largest condition number of C_k (that of the regressors squared).
    """
    u = c.values
    F = features(basis, u, c.scale)  # (batch, time, m, n_ops)
    t = np.arange(t0, u.shape[1] - 1)
    phi = np.concatenate([F[:, t - s] for s in range(lags + 1)], axis=-1)  # (batch, nt, m, (lags + 1) n_ops)
    du = np.fft.fft(u[:, t + 1] - u[:, t], axis=-1)  # (batch, nt, m)
    n = phi.shape[0] * phi.shape[1]
    C = np.einsum("btmp,btmq->mpq", phi.conj(), phi) / n
    b = np.einsum("btmp,btm->mp", phi.conj(), du) / n
    coeffs = LaggedModel(basis, lags, homogeneous=True, ridge=ridge).fit(c, t0).coeffs  # (m, p)
    coeff_diff, step_diff, cond = 0.0, 0.0, 0.0
    for q in range(C.shape[0]):
        if not np.any(C[q]):  # a mode where every feature vanishes, e.g. dropped by the filter: both give 0
            continue
        A = C[q] + ridge * np.diag(np.diag(C[q]))
        K = np.linalg.solve(A, b[q])
        coeff_diff = max(coeff_diff, rel(K, coeffs[q]))
        step_diff = max(step_diff, rel(phi[:, :, q] @ K, phi[:, :, q] @ coeffs[q]))
        cond = max(cond, np.linalg.cond(C[q]))
    return coeff_diff, step_diff, cond


def pipeline_rollout(dynamics, c, t0, t1, memory):
    """The same rollout with an EffectiveDynamics from mori_zwanzig_data (Markov, or with gamma(t))."""
    u = [c.values[:, t0]]
    for t in range(t0, t1):
        state = c.replace(values=u[-1][:, None], time=c.time[t : t + 1])
        u.append(u[-1] + dynamics.evaluate(state, memory=memory).values[:, 0])
    return np.stack(u, axis=1)


def rel(predicted, true, axis=None):
    return np.linalg.norm(predicted - true, axis=axis) / np.linalg.norm(true, axis=axis)


# ---- the comparison ----

S_MAX = 3  # largest lag; every model is fitted and scored on the steps t >= S_MAX
RIDGE = 1e-2  # relative ridge penalty of the regularised lagged models
T1 = N_TIME - 2  # last state predicted: the gamma(t) model of mori_zwanzig_data covers the steps up to N_TIME - 3


def compare(name, train, test, basis, n_levels, homogeneous):
    data = mori_zwanzig_data(train, n_levels=n_levels, basis=basis, homogeneous=homogeneous)
    t = np.arange(S_MAX, T1)  # scored steps
    print(f"\n===== {name}, homogeneous={homogeneous}  (held-out ensemble; steps t = {TIMES[S_MAX]:.2f}..{TIMES[T1 - 1]:.2f})")
    print(f"{'level':>5} {'model':>24} {'unknowns/mode':>14} {'train one-step':>15} {'test one-step':>14} "
          f"{'rollout mean':>13} {'rollout end':>12}")
    for d in data:
        c_train, c_test = coarse(train, d.level), coarse(test, d.level)
        true_steps = c_test.values[:, t + 1] - c_test.values[:, t]
        true_roll = c_test.values[:, S_MAX : T1 + 1]
        rows = {}
        dynamics = d.dynamics()
        states = c_test.replace(values=c_test.values[:, t], time=c_test.time[t])
        for label, memory in [("pipeline Markov (t=0)", False), ("pipeline gamma(t)", True)]:
            roll = pipeline_rollout(dynamics, c_test, S_MAX, T1, memory)
            rows[label] = ("-", np.nan, rel(dynamics.evaluate(states, memory=memory).values, true_steps), roll)
        m = c_test.values.shape[-1]
        for lags, ridge in [(s, 0.0) for s in range(S_MAX + 1)] + [(1, RIDGE), (S_MAX, RIDGE)]:
            model = LaggedModel(basis, lags, homogeneous, ridge).fit(c_train, S_MAX)
            train_steps = c_train.values[:, t + 1] - c_train.values[:, t]
            unknowns = (lags + 1) * len(basis) * (1 if homogeneous else m)
            rows[f"lagged S={lags}" + (f" ridge {ridge:g}" if ridge else "")] = (
                unknowns,
                rel(model.one_step(c_train, t), train_steps),
                rel(model.one_step(c_test, t), true_steps),
                model.rollout(c_test, S_MAX, T1),
            )
        for label, (unknowns, train_err, test_err, roll) in rows.items():
            per_time = rel(roll, true_roll, axis=(0, 2))
            print(f"{d.level:>5} {label:>24} {unknowns:>14} {train_err:>15.4f} {test_err:>14.4f} "
                  f"{per_time[1:].mean():>13.4g} {per_time[-1]:>12.4g}")


if __name__ == "__main__":
    train = {"burgers": ensemble(burgers, 0, N_TRAIN), "advection": ensemble(nonuniform_advection, 0, N_TRAIN)}
    test = {"burgers": ensemble(burgers, 1, N_TEST), "advection": ensemble(nonuniform_advection, 1, N_TEST)}
    print("===== lstsq fit vs the correlation equations (Burgers, homogeneous; max over modes)")
    print(f"{'basis':>20} {'level':>5} {'lags':>4} {'ridge':>6} {'coeff diff':>11} {'step diff':>10} {'cond C':>9}")
    for basis in [OperatorBasis([Identity()]), OperatorBasis([Identity(), Advection()])]:
        for level in [1, 3]:
            for lags, ridge in [(0, 0.0), (S_MAX, 0.0), (S_MAX, RIDGE)]:
                diffs = correlation_check(coarse(train["burgers"], level), basis, lags, S_MAX, ridge)
                print(f"{str(basis.names):>20} {level:>5} {lags:>4} {ridge:>6g} {diffs[0]:>11.2e} {diffs[1]:>10.2e} "
                      f"{diffs[2]:>9.2e}")
    compare("Burgers, basis [u]", train["burgers"], test["burgers"], OperatorBasis([Identity()]), 4, True)
    compare("Burgers, basis [u, -(u^2/2)_x]", train["burgers"], test["burgers"],
            OperatorBasis([Identity(), Advection()]), 4, True)
    compare("non-uniform advection, basis [u]", train["advection"], test["advection"],
            OperatorBasis([Identity()]), 3, False)
