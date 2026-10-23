"""
Stochastic analog RNN as a model of the Mori-Zwanzig orthogonal dynamics
=========================================================================

Toy system (Kac-Zwanzig heat bath): one particle in a double well, linearly
coupled to N harmonic bath oscillators. For this system the Mori-Zwanzig
equation for the resolved variables phi = (q, p) is EXACT and known in closed form:

    dq/dt = p
    dp/dt = -V'(q) - int_0^t K(t-s) p(s) ds + F(t)

    K(t) = sum_j c_j cos(w_j t)                         (memory kernel)
    F(t) = sum_j g_j [ y_j(0) cos(w_j t)
                       + v_j(0)/w_j sin(w_j t) ]        (orthogonal dynamics)

where y_j = x_j - g_j q / w_j^2 are the shifted bath coordinates and g_j = w_j sqrt(c_j).
With a thermal (canonical) bath, F is Gaussian, mean zero given phi(0)
(orthogonality), and satisfies the fluctuation-dissipation theorem
E[F(t)F(0)] = kT K(t). So we know the "ground truth" orthogonal dynamics and can
test whether a learned stochastic RNN reproduces it.

Pipeline
  1. Simulate the full (N+1)-body system; record phi(t) and the exact F(t).
  2. Train a stochastic analog RNN (saturated-linear units + injected noise)
     to GENERATE F given phi(0), using a distributional loss (energy score).
     Ablation: the same network trained with MSE (collapses to F ~ 0).
  3. Check: autocovariance vs kT*K(t), marginals, orthogonality E[F g(phi0)] = 0.
  4. Closed loop: plug the learned F into the reduced GLE and compare
     the resolved statistics with the full model.

Run:  python mz_stochastic_arnn.py         (CPU, ~15 min)
Outputs: mz_results.png, printed diagnostics.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_num_threads(2)
rng = np.random.default_rng(0)
torch.manual_seed(0)

# --------------------------------------------------------------------------
# 1. Full system
# --------------------------------------------------------------------------
kT = 0.5                      # temperature
N_BATH = 100                  # bath oscillators
dw = 0.1
w = dw * np.arange(1, N_BATH + 1)          # frequencies 0.1 ... 10
eta, wc = 1.0, 2.0                          # target K(t) ~ eta*wc*exp(-wc t)
c = (2 / np.pi) * eta * dw * wc**2 / (wc**2 + w**2)
g = w * np.sqrt(c)                          # coupling constants

def Vprime(q):                # double well V = q^4/4 - q^2/2
    return q**3 - q

def K_exact(t):
    return (c[None, :] * np.cos(np.outer(t, w))).sum(1)

DT_INT = 0.01                 # integrator step
SUB = 5                       # record every SUB steps
DT = DT_INT * SUB             # = 0.05, model time step
T_STEPS = 200                 # recorded steps (t up to 10)
t_grid = DT * np.arange(T_STEPS)

def sample_boltzmann_q(n):
    """Rejection-sample q ~ exp(-V(q)/kT) on [-3, 3]."""
    out = []
    while len(out) < n:
        q = rng.uniform(-3, 3, 4 * n)
        V = q**4 / 4 - q**2 / 2
        acc = rng.uniform(size=q.size) < np.exp(-(V + 0.25) / kT)
        out.extend(q[acc].tolist())
    return np.array(out[:n])

def simulate_full(M):
    q = sample_boltzmann_q(M)
    p = rng.normal(0, np.sqrt(kT), M)
    y = rng.normal(0, 1, (M, N_BATH)) * np.sqrt(kT) / w     # shifted bath coords
    v = rng.normal(0, np.sqrt(kT), (M, N_BATH))
    y0, v0 = y.copy(), v.copy()

    def rhs(q, p, y, v):
        return (p,
                -Vprime(q) + y @ g,
                v - np.outer(p, g / w**2),
                -(w**2) * y)

    Q, P = np.zeros((M, T_STEPS)), np.zeros((M, T_STEPS))
    for k in range(T_STEPS):
        Q[:, k], P[:, k] = q, p
        for _ in range(SUB):                      # RK4
            a = rhs(q, p, y, v)
            b = rhs(*(s + 0.5 * DT_INT * d for s, d in zip((q, p, y, v), a)))
            cc = rhs(*(s + 0.5 * DT_INT * d for s, d in zip((q, p, y, v), b)))
            d_ = rhs(*(s + DT_INT * d for s, d in zip((q, p, y, v), cc)))
            q, p, y, v = (s + DT_INT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                          for s, k1, k2, k3, k4 in zip((q, p, y, v), a, b, cc, d_))
    # exact orthogonal dynamics on the recorded grid
    cosw, sinw = np.cos(np.outer(t_grid, w)), np.sin(np.outer(t_grid, w))
    F = (y0 * g) @ cosw.T + (v0 * g / w) @ sinw.T          # (M, T)
    return Q, P, F

t0 = time.time()
M_TRAIN, M_TEST = 4000, 2000
Q, P, F = simulate_full(M_TRAIN + M_TEST)
print(f"[1] simulated full system: {Q.shape[0]} trajectories in {time.time()-t0:.1f}s")

# Sanity check of the MZ decomposition: F(t) - memory(t) must equal the bath force
Kg = K_exact(t_grid)
def memory_term(Pm):
    """int_0^t K(t-s) p(s) ds, trapezoid rule on the recorded grid."""
    T = Pm.shape[1]
    out = np.zeros_like(Pm)
    for k in range(1, T):
        wts = np.full(k + 1, DT); wts[0] = wts[-1] = DT / 2
        out[:, k] = (Pm[:, :k + 1] * (Kg[k::-1] * wts)).sum(1)
    return out
# finite-difference dp/dt vs GLE right-hand side
dpdt = (P[:200, 2:] - P[:200, :-2]) / (2 * DT)
rhs_gle = (-Vprime(Q[:200]) - memory_term(P[:200]) + F[:200])[:, 1:-1]
rel = np.abs(dpdt - rhs_gle).mean() / np.abs(dpdt).mean()
print(f"    GLE consistency check (relative residual, O(dt^2) expected): {rel:.3f}")

# --------------------------------------------------------------------------
# 2. Stochastic analog RNN generator for F
# --------------------------------------------------------------------------
class StochasticARNN(nn.Module):
    """
    h_0     = sat( A phi0 + a + S0 xi_0 )
    h_{k+1} = sat( W h_k + b + S sqrt(dt) xi_k )
    F_k     = C h_k
    sat(z)  = clamp(z, 0, 1)    <- Siegelmann's saturated-linear 'analog' unit

    The noise xi plays the role of the unresolved initial data (bath state).
    antithetic=True returns 0.5*(F[xi] - F[-xi]): an odd function of the noise,
    so E[F | phi0] = 0 holds EXACTLY -> orthogonality by construction.
    """
    def __init__(self, d_phi=2, d_h=64, d_noise=16, antithetic=True):
        super().__init__()
        self.A = nn.Linear(d_phi, d_h)
        self.S0 = nn.Linear(d_noise, d_h, bias=False)
        self.W = nn.Linear(d_h, d_h)
        self.S = nn.Linear(d_noise, d_h, bias=False)
        self.C = nn.Linear(d_h, 1, bias=False)
        self.d_noise, self.antithetic = d_noise, antithetic
        # Start every unit in the LINEAR part of the saturated-linear activation:
        # W h + b ~ 0.5 and small noise. If units start (or drift) into the
        # flat regions 0 or 1, they behave like binary coins and F becomes
        # multimodal (this is what happened in the first version of this script).
        nn.init.normal_(self.W.weight, std=0.3 / np.sqrt(d_h))
        nn.init.normal_(self.S.weight, std=0.3 / np.sqrt(d_noise))
        nn.init.normal_(self.S0.weight, std=0.1 / np.sqrt(d_noise))
        with torch.no_grad():
            self.W.weight.add_(0.9 * torch.eye(d_h))   # slow, near-linear modes
            self.W.bias.fill_(0.05)
            self.A.bias.fill_(0.5)

    def _run(self, phi0, xi0, xis, dt):
        h = torch.clamp(self.A(phi0) + self.S0(xi0), 0, 1)
        out = [self.C(h)]
        for k in range(xis.shape[0]):
            h = torch.clamp(self.W(h) + self.S(xis[k]) * dt**0.5, 0, 1)
            out.append(self.C(h))
        return torch.cat(out, -1)                      # (batch, T)

    def forward(self, phi0, T, dt=DT):
        B = phi0.shape[0]
        xi0 = torch.randn(B, self.d_noise)
        xis = torch.randn(T - 1, B, self.d_noise)
        Fp = self._run(phi0, xi0, xis, dt)
        if not self.antithetic:
            return Fp
        Fm = self._run(phi0, -xi0, -xis, dt)
        return 0.5 * (Fp - Fm)

def energy_score(samples, target):
    """samples (m, B, T), target (B, T). Proper scoring rule for distributions."""
    m = samples.shape[0]
    t1 = (samples - target[None]).norm(dim=-1).mean()
    d = (samples[:, None] - samples[None]).norm(dim=-1)        # (m, m, B)
    t2 = d.sum() / (m * (m - 1) * samples.shape[1])
    return t1 - 0.5 * t2

def multiscale_energy(samples, target):
    """Energy score on F and on its increments over several lags
    (makes the loss sensitive to the temporal correlation structure)."""
    loss = energy_score(samples, target)
    # marginal term: 1-D energy score (= CRPS) at every time, averaged.
    # Trajectory-level energy scores mostly see covariance; this term makes
    # the loss sensitive to the SHAPE of the marginal distribution of F(t).
    m = samples.shape[0]
    crps = energy_score(samples.reshape(m, -1, 1), target.reshape(-1, 1))
    loss = loss + np.sqrt(samples.shape[-1]) * crps
    for lag in (1, 4, 16):
        loss = loss + energy_score(samples[..., lag:] - samples[..., :-lag],
                                   target[..., lag:] - target[..., :-lag])
    return loss

T_TRAIN = 120                                         # train horizon (t = 6)
phi0 = torch.tensor(np.stack([Q[:, 0], P[:, 0]], 1), dtype=torch.float32)
Ft = torch.tensor(F, dtype=torch.float32)
tr, te = slice(0, M_TRAIN), slice(M_TRAIN, None)

def train(model, loss_kind, iters, batch=64, m=8, lr=3e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    for it in range(iters):
        idx = torch.randint(0, M_TRAIN, (batch,))
        x, y = phi0[tr][idx], Ft[tr][idx, :T_TRAIN]
        if loss_kind == "energy":
            s = torch.stack([model(x, T_TRAIN) for _ in range(m)])
            loss = multiscale_energy(s, y)
        else:                                          # naive MSE on one sample
            loss = ((model(x, T_TRAIN) - y) ** 2).mean()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if it % 250 == 0 or it == iters - 1:
            print(f"    [{loss_kind}] iter {it:5d}  loss {loss.item():.4f}")
    return model

print("[2] training stochastic ARNN with energy score ...")
t0 = time.time()
model_es = train(StochasticARNN(), "energy", iters=1500)
print(f"    done in {time.time()-t0:.0f}s")
print("[2b] training the same network with MSE (ablation) ...")
model_mse = train(StochasticARNN(), "mse", iters=600)

# --------------------------------------------------------------------------
# 3. Statistical checks
# --------------------------------------------------------------------------
@torch.no_grad()
def sample_F(model, x, T):
    return model(x, T).numpy()

x_te = phi0[te]
F_true = F[M_TRAIN:]
F_es = sample_F(model_es, x_te, T_STEPS)
F_mse = sample_F(model_mse, x_te, T_STEPS)

def autocov(Fm, lags=80):
    """Stationary autocovariance E[F(s+t)F(s)], averaged over s."""
    return np.array([(Fm[:, l:] * Fm[:, :Fm.shape[1] - l]).mean() for l in range(lags)])

LAGS = 80
ac_true, ac_es, ac_mse = autocov(F_true, LAGS), autocov(F_es, LAGS), autocov(F_mse, LAGS)
ac_fdt = kT * Kg[:LAGS]

def orth(Fm):
    """Correlations of F(t) with functions of phi(0): should be 0."""
    q0, p0 = Q[M_TRAIN:, 0], P[M_TRAIN:, 0]
    tests = {"q0": q0, "p0": p0, "q0^2": q0**2, "q0^3": q0**3}
    return {k: np.mean([np.corrcoef(Fm[:, j], v)[0, 1] for j in range(0, T_STEPS, 10)])
            for k, v in tests.items()}

@torch.no_grad()
def saturation_fraction(model, x, T=T_STEPS):
    """Fraction of hidden-unit activations stuck exactly at 0 or 1."""
    h = torch.clamp(model.A(x) + model.S0(torch.randn(x.shape[0], model.d_noise)), 0, 1)
    sat = []
    for _ in range(T - 1):
        h = torch.clamp(model.W(h) + model.S(torch.randn(x.shape[0], model.d_noise)) * DT**0.5, 0, 1)
        sat.append(((h == 0) | (h == 1)).float().mean().item())
    return float(np.mean(sat))

print("[3] diagnostics on held-out trajectories")
print(f"    saturated hidden units: energy-ARNN {saturation_fraction(model_es, x_te):.2%}")
print(f"    Var F(0): true {F_true[:,0].var():.3f} | FDT kT*K(0) {kT*Kg[0]:.3f} | "
      f"energy-ARNN {F_es[:,0].var():.3f} | MSE-ARNN {F_mse[:,0].var():.4f}")
err = lambda a: np.abs(a - ac_fdt).max() / ac_fdt[0]
print(f"    max |autocov - kT K| / kT K(0): true {err(ac_true):.3f} | "
      f"energy {err(ac_es):.3f} | MSE {err(ac_mse):.3f}")
print(f"    orthogonality corr(F, g(phi0)), true:   {orth(F_true)}")
print(f"    orthogonality corr(F, g(phi0)), energy: {orth(F_es)}")

# --------------------------------------------------------------------------
# 4. Closed-loop reduced model (GLE) driven by each noise model
# --------------------------------------------------------------------------
def run_gle(q0, p0, Fn):
    """Semi-implicit Euler for dq=p, dp=-V'(q)-memory+F, memory by trapezoid."""
    M, T = Fn.shape
    q, p = np.zeros((M, T)), np.zeros((M, T))
    q[:, 0], p[:, 0] = q0, p0
    for k in range(T - 1):
        if k == 0:
            mem = 0.0
        else:
            wts = np.full(k + 1, DT); wts[0] = wts[-1] = DT / 2
            mem = (p[:, :k + 1] * (Kg[k::-1] * wts)).sum(1)
        p[:, k + 1] = p[:, k] + DT * (-Vprime(q[:, k]) - mem + Fn[:, k])
        q[:, k + 1] = q[:, k] + DT * p[:, k + 1]
    return q, p

q0, p0 = Q[M_TRAIN:, 0], P[M_TRAIN:, 0]
runs = {
    "full model": (Q[M_TRAIN:], P[M_TRAIN:]),
    "GLE + exact F": run_gle(q0, p0, F_true),
    "GLE + energy-ARNN": run_gle(q0, p0, F_es),
    "GLE + MSE-ARNN": run_gle(q0, p0, F_mse),
}
print("[4] closed loop: <p^2> at t=10 (equipartition says kT = 0.5)")
for k, (qq, pp) in runs.items():
    print(f"    {k:20s} <p^2>(t=10) = {(pp[:, -1]**2).mean():.3f}   "
          f"<q^2>(t=10) = {(qq[:, -1]**2).mean():.3f}")

# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"
C_TRUE, C_ES, C_MSE, C_ALT = INK, "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "lines.linewidth": 2, "legend.frameon": False})
fig, ax = plt.subplots(2, 3, figsize=(13, 7.5))

a = ax[0, 0]
for i in range(3):
    a.plot(t_grid, F_true[i], color=C_TRUE, lw=1.2, alpha=0.8,
           label="exact F (3 samples)" if i == 0 else None)
    a.plot(t_grid, F_es[i], color=C_ES, lw=1.2, alpha=0.8,
           label="energy-ARNN (3 samples)" if i == 0 else None)
a.plot(t_grid, F_mse[0], color=C_MSE, lw=2, label="MSE-ARNN")
a.axvline(DT * T_TRAIN, color=MUTED, ls=":", lw=1)
a.text(DT * T_TRAIN + 0.1, a.get_ylim()[1] * 0.85, "train horizon", color=MUTED, fontsize=8)
a.set(title="Sample paths of the orthogonal dynamics F(t)", xlabel="t", ylabel="F")
a.legend(fontsize=8, loc="lower left")

a = ax[0, 1]
lt = t_grid[:LAGS]
a.plot(lt, ac_fdt, color=C_TRUE, ls="--", label="kT·K(t)  (FDT)")
a.plot(lt, ac_true, color=C_ALT, label="exact F, empirical")
a.plot(lt, ac_es, color=C_ES, label="energy-ARNN")
a.plot(lt, ac_mse, color=C_MSE, label="MSE-ARNN")
a.set(title="Autocovariance  E[F(s+t) F(s)]", xlabel="lag t", ylabel="covariance")
a.legend(fontsize=8)

a = ax[0, 2]
bins = np.linspace(-4, 4, 50)
j = 100
a.hist(F_true[:, j], bins, density=True, color=C_TRUE, alpha=0.25, label="exact F")
a.hist(F_es[:, j], bins, density=True, histtype="step", color=C_ES, lw=2, label="energy-ARNN")
xs = np.linspace(-4, 4, 200); s2 = kT * Kg[0]
a.plot(xs, np.exp(-xs**2 / (2 * s2)) / np.sqrt(2 * np.pi * s2), color=MUTED, ls="--",
       lw=1.5, label="N(0, kT·K(0))")
a.set(title=f"Marginal of F at t = {t_grid[j]:.0f}", xlabel="F", ylabel="density")
a.legend(fontsize=8)

a = ax[1, 0]
o_t, o_e, o_m = orth(F_true), orth(F_es), orth(F_mse)
keys = list(o_t); xk = np.arange(len(keys)); bw = 0.26
a.bar(xk - bw, [o_t[k] for k in keys], bw * 0.92, color=C_ALT, label="exact F")
a.bar(xk, [o_e[k] for k in keys], bw * 0.92, color=C_ES, label="energy-ARNN")
a.bar(xk + bw, [o_m[k] for k in keys], bw * 0.92, color=C_MSE, label="MSE-ARNN")
a.axhspan(-2 / np.sqrt(M_TEST), 2 / np.sqrt(M_TEST), color=GRID, alpha=0.8, lw=0)
a.set_xticks(xk, [f"g = {k}" for k in keys])
a.set(title="Orthogonality: corr(F(t), g(φ(0)))  ≈ 0", ylabel="correlation", ylim=(-0.15, 0.15))
a.legend(fontsize=8)

a = ax[1, 1]
cols = {"full model": C_TRUE, "GLE + exact F": C_ALT,
        "GLE + energy-ARNN": C_ES, "GLE + MSE-ARNN": C_MSE}
for k, (qq, pp) in runs.items():
    a.plot(t_grid, (pp**2).mean(0), color=cols[k], label=k,
           ls="--" if k == "full model" else "-")
a.axhline(kT, color=MUTED, lw=1, ls=":")
a.set(title="Closed loop: kinetic energy ⟨p²⟩(t)", xlabel="t", ylabel="⟨p²⟩", ylim=(0, 0.7))
a.legend(fontsize=8, loc="lower left")

a = ax[1, 2]
def acf_p(pp, lags=LAGS):
    return np.array([(pp[:, 40 + l] * pp[:, 40]).mean() for l in range(lags)])
for k, (qq, pp) in runs.items():
    a.plot(lt, acf_p(pp), color=cols[k], label=k, ls="--" if k == "full model" else "-")
a.set(title="Closed loop: velocity autocorrelation", xlabel="lag t", ylabel="⟨p(s+t) p(s)⟩")
a.legend(fontsize=8)

fig.suptitle("Stochastic analog RNN as a model of Mori–Zwanzig orthogonal dynamics "
             "(Kac–Zwanzig heat bath, double-well particle)", fontsize=11, color=INK)
fig.tight_layout()
fig.savefig("mz_results.png", dpi=150)
print("saved mz_results.png")
