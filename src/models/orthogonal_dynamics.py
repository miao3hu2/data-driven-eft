"""
The network of the predictable part of the noise of a Mori-Zwanzig closure (mz_noise.MeanNoise).

W is fixed by the full initial state, but the closure knows only part of that state, x. So W splits in two:

    W(n) = m(x, n) + eta(n)

m(x, n) is the mean of W over the unresolved initial data consistent with x: the best guess of W(n) given x alone. This
network computes it, and holds no randomness. eta has zero mean, and is random because the unresolved data is unknown:
that is the stochastic RNN's (stochastic_rnn.py).

    h(0)    = tanh(E(x))
    h(n+1)  = a h(n) + (1 - a) tanh(F(h(n)))        0 < a < 1
    m(n+1)  = G(h(n+1))

E, F and G are circular convolutions and h is a set of fields in space. x enters only through h(0), and the step only
through the number of updates of h.

The structure guarantees, whatever the weights:

    no feedback from the simulation  h evolves on its own, so m cannot depend on the simulated u after the initial states
    a bounded output                 |h| <= 1 at all times, so m cannot diverge however long the rollout
    the symmetries of the field      shifting x in space shifts m by the same amount, and m is real in space, so
                                     m_-k = conj(m_k) and m_0 is real

It does not guarantee that m is the true conditional mean, which is only what the training aims for. m may also partly
correct errors of the memory kernels, which has not been tested.
"""

import torch
import torch.nn as nn


def _convolutions(channels_in: int, hidden: int, channels_out: int, depth: int = 3) -> nn.Sequential:
    """Circular convolutions along x, (batch, channels_in, n_x) -> (batch, channels_out, n_x)."""
    layers: list[nn.Module] = []
    width = channels_in
    for _ in range(depth - 1):
        layers += [nn.Conv1d(width, hidden, 3, padding=1, padding_mode="circular"), nn.GELU()]
        width = hidden
    layers.append(nn.Conv1d(width, channels_out, 3, padding=1, padding_mode="circular"))
    return nn.Sequential(*layers)


class MeanNetwork(nn.Module):
    """The mean m of W as a deterministic function of the known initial states x and of the step, see the module docstring."""

    def __init__(self, n_inputs: int, d_h: int = 8, hidden: int = 16, seed: int = 0):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.encoder = _convolutions(n_inputs, hidden, d_h)
            self.flow = _convolutions(d_h, hidden, d_h, depth=2)
            self.decoder = _convolutions(d_h, hidden, 1, depth=2)
        self.logit_a = nn.Parameter(torch.full((d_h,), 2.0))  # a = 0.88: starts slow, as the noise is correlated in time
        self.double()

    def initial_state(self, x: torch.Tensor) -> torch.Tensor:
        """h(0) from the known initial states x (batch, n_inputs, n_x)."""
        return torch.tanh(self.encoder(x))

    def evolve(self, h: torch.Tensor) -> torch.Tensor:
        """h(n) -> h(n+1): autonomous and bounded, a convex combination of values in [-1, 1]."""
        a = torch.sigmoid(self.logit_a)[:, None]
        return a * h + (1 - a) * torch.tanh(self.flow(h))

    def forward(self, x: torch.Tensor, n_steps: int) -> torch.Tensor:
        """The path of the mean, at steps 1, ..., n_steps, in standardized units, as real fields (batch, n_steps, n_x)."""
        h, path = self.initial_state(x), []
        for _ in range(n_steps):
            h = self.evolve(h)
            path.append(self.decoder(h).squeeze(1))
        return torch.stack(path, dim=1)
