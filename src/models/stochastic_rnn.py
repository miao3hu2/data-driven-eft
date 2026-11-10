"""
A stochastic recurrent network for the random part of the noise of a Mori-Zwanzig effective dynamics on non-stationary trajectories,
one Fourier mode at a time, with a distribution that changes in time.

The noise splits as W(n) = m(x, n) + eta(n), where m is its mean given the known initial states x (orthogonal_dynamics.py).
eta has zero mean, and is random because the unresolved initial data is unknown. This network models eta:

    h_k(n+1)    = A_k(n) h_k(n) + sigma_k(n) S_k xi_k(n),     h_k(0) = S0_k xi0_k
    eta_k(n+1)  = c_k^H h_k(n+1) + sigma_k(n) sqrt(r_k) eps_k(n)

    A_k(n) = sum_j A_kj tau^j,   log sigma_k(n) = sum_j s_kj tau^j,   tau = min(n / (T - 1), 1)

n counts the steps since the initial time of the effective dynamics and T is the length of the training paths.

xi0, xi and eps are standard Gaussians, circular complex ones, real ones on the real modes.
"""

import math

import torch
import torch.nn as nn

Latent = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def complex_normal(shape, generator=None) -> torch.Tensor:
    """Standard circular complex Gaussian: E|z|^2 = 1."""
    real = torch.randn(*shape, 2, generator=generator, dtype=torch.float64) / math.sqrt(2)
    return torch.view_as_complex(real)


def _apply(matrix: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Matrix times vector, over the leading axes: (..., a, b), (..., b) -> (..., a)."""
    return (matrix @ x[..., None]).squeeze(-1)


class StochasticRNN(nn.Module):

    real: torch.Tensor  # (K,) whether a mode is real; a buffer, set in __init__
    active: torch.Tensor  # (K,) whether a mode has any noise; a buffer, set in __init__

    def __init__(self, n_modes: int, n_steps: int, real_modes: torch.Tensor | None = None, active: torch.Tensor | None = None, d_h: int = 2, degree_sigma: int = 3, degree_A: int = 2, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        K, d = n_modes, d_h
        self.d_h, self.n_steps = d, n_steps
        eye = torch.eye(d, dtype=torch.complex128).expand(K, d, d)
        A = torch.zeros(degree_A + 1, K, d, d, dtype=torch.complex128)
        A[0] = 0.9 * eye
        self.A = nn.Parameter(A)  # (degree_A + 1, K, d, d), starting constant in time
        self.log_sigma = nn.Parameter(torch.zeros(degree_sigma + 1, K, dtype=torch.float64))
        self.S = nn.Parameter(0.3 * eye.clone())
        self.S0 = nn.Parameter(eye.clone())
        self.c = nn.Parameter(0.3 * complex_normal((K, d), g))
        self.log_r = nn.Parameter(torch.full((K,), -1.0, dtype=torch.float64))
        real = torch.zeros(K, dtype=torch.bool) if real_modes is None else torch.as_tensor(real_modes, dtype=torch.bool)
        self.register_buffer("real", real)
        self.register_buffer("active", torch.ones(K, dtype=torch.bool) if active is None else torch.as_tensor(active, dtype=torch.bool))

    # ---- the parameters as used: real on the real modes, A contracting, functions of time ----

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        mask = self.real.view(-1, *([1] * (x.dim() - 1)))
        return torch.where(mask, x.real.to(x.dtype), x)

    def _tau(self, n: int) -> float:
        return min(n / max(self.n_steps - 1, 1), 1.0)

    def A_at(self, n: int) -> torch.Tensor:
        """A(n), (K, d, d), with |A(n)| <= 0.999 per mode so that the hidden state cannot grow without bound."""
        tau = self._tau(n)
        A = self._project(sum(tau**j * self.A[j] for j in range(len(self.A))))  # type: ignore[arg-type]
        if self.d_h:
            norm = torch.linalg.matrix_norm(A, ord=2)
            A = A * torch.clamp(0.999 / (norm + 1e-12), max=1.0)[:, None, None]
        return A

    def sigma_at(self, n: int) -> torch.Tensor:
        """sigma(n), (K,)."""
        tau = self._tau(n)
        return torch.exp(sum(tau**j * self.log_sigma[j] for j in range(len(self.log_sigma))))  # type: ignore[arg-type]

    # ---- the exact likelihood ----

    def nll(self, eta: torch.Tensor) -> torch.Tensor:
        """Mean negative log-likelihood per active mode and step of a path of eta (batch, T, K)."""
        S, S0, c, r = self._project(self.S), self._project(self.S0), self._project(self.c), self.log_r.exp()
        batch, T, K = eta.shape
        h_mean = torch.zeros(batch, K, self.d_h, dtype=torch.complex128)
        P = (S0 @ S0.mH).expand(batch, K, self.d_h, self.d_h)
        SS = S @ S.mH
        total = torch.zeros((), dtype=torch.float64)
        for n in range(T):
            A, sigma2 = self.A_at(n), self.sigma_at(n) ** 2
            h_mean = _apply(A, h_mean)  # predict h(n+1)
            P = A @ P @ A.mH + sigma2[:, None, None] * SS
            Pc = _apply(P, c)  # (batch, K, d)
            s = (c.conj() * Pc).sum(-1).real + sigma2 * r  # variance of eta(n+1) given the past
            e = eta[:, n] - (c.conj() * h_mean).sum(-1)
            complex_nll = torch.log(math.pi * s) + e.abs() ** 2 / s
            real_nll = 0.5 * torch.log(2 * math.pi * s) + e.real**2 / (2 * s)
            total = total + torch.where(self.real, real_nll, complex_nll)[:, self.active].mean()
            gain = Pc / s[..., None]  # condition h(n+1) on eta(n+1)
            h_mean = h_mean + gain * e[..., None]
            P = P - gain[..., :, None] * gain.conj()[..., None, :] * s[..., None, None]
        return total / T

    # ---- generating the noise: one draw, then a deterministic path ----

    def _normal(self, shape: tuple[int, ...], generator, trailing: int = 0) -> torch.Tensor:
        """Standard Gaussians (batch, K, *trailing axes), real on the real modes."""
        z = complex_normal(shape, generator)
        mask = self.real.view(1, -1, *([1] * trailing))
        return torch.where(mask, torch.randn(*shape, generator=generator, dtype=torch.float64).to(z.dtype), z)

    def latent(self, batch: int, n_steps: int, generator: torch.Generator | None = None) -> Latent:
        """All of the randomness of a path, drawn at the initial time: xi0 (batch, K, d), xi (batch, n_steps, K, d), eps (batch, n_steps, K)."""
        K = len(self.real)
        xi0 = self._normal((batch, K, self.d_h), generator, trailing=1)
        xi = torch.stack([self._normal((batch, K, self.d_h), generator, trailing=1) for _ in range(n_steps)], dim=1)
        eps = torch.stack([self._normal((batch, K), generator) for _ in range(n_steps)], dim=1)
        return xi0, xi, eps

    @torch.no_grad()
    def path(self, z: Latent) -> torch.Tensor:
        """eta(1..n_steps) (batch, n_steps, K), a deterministic function of the latent; 0 on the modes without noise."""
        xi0, xi, eps = z
        S, S0, c, r = self._project(self.S), self._project(self.S0), self._project(self.c), self.log_r.exp()
        h, path = _apply(S0, xi0), []
        for n in range(xi.shape[1]):
            sigma = self.sigma_at(n)
            h = _apply(self.A_at(n), h) + sigma[:, None] * _apply(S, xi[:, n])
            path.append((c.conj() * h).sum(-1) + sigma * r.sqrt() * eps[:, n])
        return torch.stack(path, dim=1) * self.active
