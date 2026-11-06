from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view

from core.data.base import Field
from core.inference.mori_kernels import MemoryKernels, observables
from models.orthogonal_dynamics import MeanNetwork
from models.stochastic_rnn import StochasticRNN


def _symmetrize(a: np.ndarray) -> np.ndarray:
    """The average of each mode with its opposite, for an array (K,) in FFT order: a_k = a_-k."""
    return (a + a[-np.arange(len(a)) % len(a)]) / 2


def orthogonal_noise(u: Field, kernels: MemoryKernels) -> tuple[np.ndarray, np.ndarray]:
    """The observables g (batch, time, K, M) of a coarse field and the noise W of u (batch, time - n_lags, K): W[:, s] is the noise of the step to time n_lags + s."""
    g = observables(u, kernels.basis)
    # (batch, time - n_lags, K, M, n_lags): g(s + w), oldest first
    windows = sliding_window_view(g[:, :-1], kernels.n_lags, axis=1)
    # the u row of sum_l Omega^(l) g(n-l), the kernels' prediction of the next u
    drift = np.einsum("lkj,bskjl->bsk", kernels.kernels[::-1, :, 0, :], windows, optimize=True)
    return g, g[:, kernels.n_lags:, :, 0] - drift


@dataclass
class Standardizer:
    """The units of the network: its inputs and outputs have unit variance, mode by mode. W_k = noise_std_k * fft(y)_k."""

    modes: np.ndarray  # (K+,) indices of the modes k >= 0 among the kernels' wavevectors
    active: np.ndarray  # (K+,) whether a mode has any noise: k = 0 has none when the mean of u is conserved
    noise_std: np.ndarray  # (K,) symmetric in k, so that a real field gives W_-k = conj(W_k)
    step_std: np.ndarray  # (T, K+) of the standardized noise per step: the weights of the loss, as W decays quickly
    input_std: np.ndarray  # (M,)

    @classmethod
    def from_data(cls, kernels: MemoryKernels, g: np.ndarray, W: np.ndarray) -> Standardizer:
        k = kernels.wavevectors[:, 0]
        n = len(k)
        assert kernels.wavevectors.shape[1] == 1 and np.allclose(k, k[1] * n * np.fft.fftfreq(n)), "1D, in FFT order"
        modes = np.flatnonzero(k >= 0)
        std = np.sqrt(np.mean(np.abs(W) ** 2, axis=(0, 1)))
        noise_std = _symmetrize(std) + 1e-30
        return cls(modes=modes,
                   active=noise_std[modes] > 1e-8 * noise_std.max(),
                   noise_std=noise_std,
                   step_std=np.sqrt(np.mean(np.abs(W[..., modes] / noise_std[modes]) ** 2, axis=0)) + 1e-30,
                   input_std=np.sqrt(np.mean(np.abs(g) ** 2, axis=(0, 1, 2))) + 1e-30)

    def initial_states(self, g: np.ndarray) -> torch.Tensor:
        """The known initial data of the network: the first observables g (batch, n_lags, K, M) as real standardized fields on the grid, (batch, n_lags * M, n_x)."""
        x = np.fft.ifft(g / self.input_std, axis=2, norm="ortho").real
        batch, n_lags, n_x, n_observables = x.shape
        return torch.as_tensor(x.transpose(0, 2, 1, 3).reshape(batch, n_x, n_lags * n_observables).transpose(0, 2, 1))

    def to_modes(self, y: torch.Tensor) -> torch.Tensor:
        """Real standardized noise fields (..., n_x) -> standardized noise modes (..., K), conjugate symmetric."""
        return torch.fft.fft(y, dim=-1, norm="ortho")


@dataclass
class MeanNoise:

    nets: list[MeanNetwork]
    scale: Standardizer
    shrink: np.ndarray  # measures how much networks' predictions need to be shrunk to match the actual mean of the noise

    def predict(self, g: np.ndarray, n_steps: int) -> np.ndarray:
        x = self.scale.initial_states(g)
        # the prediction is the average of the predictions of all the MeanNetworks in self.nets
        with torch.no_grad():
            y = torch.stack([net(x, n_steps) for net in self.nets]).mean(dim=0)
        return self.scale.to_modes(y).numpy() * self.scale.noise_std * self.shrink


def fit_mean(g: np.ndarray, W: np.ndarray, kernels: MemoryKernels, folds: int = 5, d_h: int = 8, hidden: int = 16, iters: int = 500, lr: float = 2e-3, weight_decay: float = 1e-2, seed: int = 0, log=None) -> tuple[MeanNoise, np.ndarray]:
    """
    Fit m in W(n) = m(x, n) + eta(n) by cross-fitting, from the observables g and the noise W of orthogonal_noise.

    The trajectories are split into random groups (# = folds), and one MeanNetwork is trained per group, on the trajectories in the other groups, and then predicts the trajectories in its own group.
    """
    scale = Standardizer.from_data(kernels, g, W)
    x = scale.initial_states(g[:, :kernels.n_lags])
    batch, n_steps, _ = W.shape
    modes = scale.modes[scale.active]
    weights = torch.as_tensor(1 / scale.step_std[:, scale.active])
    target = torch.as_tensor(W[..., modes] / scale.noise_std[modes], dtype=torch.complex128) * weights
    # Split the trajectories into equal random groups (folds). fold_of[i] is the group number of trajectory i.
    fold_of = np.random.default_rng(seed).permutation(batch) % folds
    nets, held_out = [], np.zeros_like(W)
    for fold in range(folds):
        train = torch.as_tensor(fold_of != fold)
        net = MeanNetwork(x.shape[1], d_h, hidden, seed + fold)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
        for _ in range(iters):
            y = scale.to_modes(net(x[train], n_steps))[..., modes] * weights
            loss = (y - target[train]).abs().pow(2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
        # predict the group this network did not train on
        with torch.no_grad():
            held_out[fold_of == fold] = scale.to_modes(net(x[~train], n_steps)).numpy() * scale.noise_std
        nets.append(net)
        if log is not None:
            log(f"    fold {fold}: training loss {loss.item():.4f}")
    # to compute shrink, which measures how much networks' predictions need to be shrunk to match the actual mean of the noise. Defined per mode k as
    #     shrink_k = clip(a_k, 0, 1),   a_k = argmin_a sum_{i,n} |W_k^i(n) - a p_k^i(n)|^2 = N_k / D_k
    #     N_k = sum_{i,n} Re[W_k^i(n) conj(p_k^i(n))],   D_k = sum_{i,n} |p_k^i(n)|^2
    # with i the trajectory, n the step and p = held_out. N and D are averaged between k and -k, so that shrink_k = shrink_-k
    numerator = _symmetrize(np.real(W * held_out.conj()).sum(axis=(0, 1)))
    denominator = _symmetrize((np.abs(held_out) ** 2).sum(axis=(0, 1))) + 1e-300
    shrink = np.clip(numerator / denominator, 0.0, 1.0)
    return MeanNoise(nets, scale, shrink), W - shrink * held_out


@dataclass
class RNNNoise:

    kernels: MemoryKernels
    rnn: StochasticRNN
    scale: Standardizer
    mean: MeanNoise  # the deterministic part of W: the RNN generates what it leaves
    losses: list[float] = field(default_factory=list)

    def sample(self, g: np.ndarray, n_steps: int, generator: torch.Generator | None = None) -> np.ndarray:
        """The path of the noise from the first observables g (batch, n_lags, K, M) and one draw per trajectory: W (batch,
        n_steps, K), the mean plus the RNN. It is fixed before the simulation starts."""
        w = self.rnn.path(self.rnn.latent(g.shape[0], n_steps, generator)).numpy()
        k = self.kernels.wavevectors[:, 0]
        full = np.zeros((*w.shape[:2], len(k)), dtype=complex)
        full[..., self.scale.modes] = w * self.scale.noise_std[self.scale.modes]
        negative = np.flatnonzero(k < 0)
        full[..., negative] = full[..., -negative % len(k)].conj()
        return self.mean.predict(g, n_steps) + full


def fit_rnn_noise(u: Field, kernels: MemoryKernels, d_h: int = 2, degree_sigma: int = 3, degree_A: int = 2, iters: int = 1500, lr: float = 2e-2, seed: int = 0, log=None, mean_options: dict | None = None) -> RNNNoise:
    """The noise of u (the training field at the closure's level) under the given kernels: first its mean (fit_mean), then
    the stochastic RNN on the residuals of trajectories the mean did not see, by maximum
    likelihood with full-batch Adam and a cosine learning rate."""
    g, W = orthogonal_noise(u, kernels)
    mean, eta = fit_mean(g, W, kernels, seed=seed, log=log, **(mean_options or {}))
    scale = Standardizer.from_data(kernels, g, eta)
    modes = scale.modes
    target = torch.as_tensor(eta[..., modes] / scale.noise_std[modes], dtype=torch.complex128)
    real = torch.as_tensor(kernels.wavevectors[modes, 0] == 0)
    rnn = StochasticRNN(len(modes), eta.shape[1], real, torch.as_tensor(scale.active), d_h, degree_sigma, degree_A, seed)
    opt = torch.optim.Adam(rnn.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    losses = []
    for it in range(iters):
        loss = rnn.nll(target)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(rnn.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())
        if log is not None and (it % 500 == 0 or it == iters - 1):
            log(f"    iter {it:5d}  nll {loss.item():+.4f}")
    return RNNNoise(kernels, rnn, scale, mean, losses)


def simulate(u: Field, kernels: MemoryKernels, noise: RNNNoise | None = None, n_samples: int = 1, seed: int = 0, n_time: int | None = None) -> np.ndarray:
    assert u.time is not None, "the trajectories have a time axis"
    length: int = u.values.shape[1] if n_time is None else n_time
    n_lags, batch = kernels.n_lags, u.values.shape[0]
    n_samples = 1 if noise is None else n_samples
    path = np.repeat(u.values[None, :, :n_lags], n_samples, axis=0).reshape(n_samples * batch, n_lags, -1)
    dt = u.time[1] - u.time[0]
    noise_path = None
    if noise is not None:  # drawn once, before the simulation: nothing below can change it
        g0 = observables(u.replace(values=path, time=dt * np.arange(n_lags)), kernels.basis)
        noise_path = noise.sample(g0, length - n_lags, torch.Generator().manual_seed(seed))
    with np.errstate(all="ignore"):
        for t in range(n_lags, length):
            window = u.replace(values=path[:, -n_lags:], time=dt * np.arange(t - n_lags, t))
            g = observables(window, kernels.basis)  # (n_samples * batch, n_lags, K, M)
            uk = kernels.predict(g)[..., 0]
            if noise_path is not None:
                uk = uk + noise_path[:, t - n_lags]
            new = u.replace(values=uk, time=None, domain="fourier").to_real().values
            path = np.concatenate([path, new[:, None]], axis=1)
    return path.reshape(n_samples, batch, length, -1)


def energy_score(samples: np.ndarray, truth: np.ndarray) -> float:
    """The energy score E|X - y| - E|X - X'|/2 of an ensemble (n_samples, batch, n_x) at each truth (batch, n_x), averaged
    over the batch and relative to |y|: a proper score of the whole forecast distribution, lower is better."""
    n = len(samples)
    skill = np.linalg.norm(samples - truth, axis=-1).mean(axis=0)
    spread = sum(np.linalg.norm(samples[i] - samples[j], axis=-1) for i in range(n) for j in range(n)) / (2 * n * n)
    return float(np.mean((skill - spread) / np.linalg.norm(truth, axis=-1)))


def spectrum(u: np.ndarray) -> np.ndarray:
    """Ensemble mean power per Fourier mode and time, (..., time, n_x) -> (time, n_x), averaged over the leading axes."""
    power = np.abs(np.fft.fft(u, axis=-1, norm="forward")) ** 2
    return power.reshape(-1, *power.shape[-2:]).mean(axis=0)


def evaluate(u: Field, kernels: MemoryKernels, noise: dict[str, RNNNoise], n_samples: int = 16, seed: int = 0) -> dict[str, dict[str, float]]:
    """Statistics of the closure without noise (no_noise) and of the stochastic closure with each noise model on the held-out
    trajectories u, over the predicted times:

        error           relative L2 error of the ensemble mean, as KernelEnvironment.rollout_error
        energy_score    of the ensemble, relative to |u| (for one trajectory without noise: its relative error)
        spread_skill    ensemble spread over the error of its mean: 1 for a calibrated ensemble, 0 without noise
        spectrum_error  mean |log(simulated power / true power)| over the modes and times: energy lost or gained
        blowups         fraction of trajectories that diverge
    """
    n_lags, true = kernels.n_lags, u.values
    truth_spectrum = spectrum(true)[n_lags:]
    keep = truth_spectrum > 1e-12 * truth_spectrum.max()
    out = {}
    for name, model in {"no_noise": None, **noise}.items():
        paths = simulate(u, kernels, model, n_samples, seed)[:, :, n_lags:]  # (S, batch, T, n_x)
        finite = np.all(np.isfinite(paths) & (np.abs(paths) < 1e6), axis=(0, 2, 3))
        p, y = paths[:, finite], true[finite, n_lags:]
        mean = p.mean(axis=0)
        # pooled over the trajectories at each time, then averaged over the times, as KernelEnvironment.rollout_error
        norm = np.linalg.norm(y, axis=(0, 2))
        error = np.linalg.norm(mean - y, axis=(0, 2)) / norm
        spread = np.sqrt(np.mean(np.sum((p - mean) ** 2, axis=(1, 3)), axis=0)) / norm
        scores = [energy_score(p[:, :, t], y[:, t]) for t in range(y.shape[1])]
        sim_spectrum = spectrum(p)
        out[name] = {
            "error": float(error.mean()),
            "energy_score": float(np.mean(scores)),
            "spread_skill": float(spread.mean() / error.mean()),
            "spectrum_error": float(np.mean(np.abs(np.log(sim_spectrum[keep] / truth_spectrum[keep])))),
            "blowups": float(1 - finite.mean()),
        }
    return out
