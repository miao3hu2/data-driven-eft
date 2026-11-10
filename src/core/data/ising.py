from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from core.data.base import Field

# Square lattice Ising model in 2D with periodic boundary conditions
@dataclass
class Ising2D:
    L: int = 32
    beta: float = 0.44
    J: float = 1.0
    seed: int | None = None

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def random_state(self) -> np.ndarray:
        return self.rng.choice(
            [-1, 1],
            size=(self.L, self.L),
        )

    def energy_change(
        self,
        spins: np.ndarray,
        i: int,
        j: int,
    ) -> float:

        L = self.L

        nn = (
            spins[(i + 1) % L, j]
            + spins[(i - 1) % L, j]
            + spins[i, (j + 1) % L]
            + spins[i, (j - 1) % L]
        )

        return 2.0 * self.J * spins[i, j] * nn

    def sweep(self, spins: np.ndarray) -> None:

        for _ in range(self.L * self.L):

            i = int(self.rng.integers(self.L))
            j = int(self.rng.integers(self.L))

            dE = self.energy_change(spins, i, j)

            if dE <= 0:
                spins[i, j] *= -1

            elif self.rng.random() < np.exp(-self.beta * dE):
                spins[i, j] *= -1

    def sample(
        self,
        n_samples: int = 100,
        burn_in: int = 1000,
        thinning: int = 10,
    ) -> Field:

        spins = self.random_state()

        for _ in range(burn_in):
            self.sweep(spins)

        samples = []

        for _ in range(n_samples):

            for _ in range(thinning):
                self.sweep(spins)

            samples.append(spins.copy())

        values = np.stack(samples)

        return Field(
            values=values,
            scale=1.0,
            batched=True,
            metadata={
                "system": "2d_ising",
                "beta": self.beta,
                "J": self.J,
            },
        )

    def sample_clusters(self, n_samples: int = 100, sweeps: int = 100) -> Field:
        """n_samples independent configurations: n_samples chains, all started at random and run in parallel for
        `sweeps` Swendsen-Wang sweeps, the last state of each chain is one sample.

        A sweep puts a bond between every pair of equal nearest neighbours with probability 1 - exp(-2 beta J), finds
        the clusters these bonds connect, and flips each cluster with probability 1/2. It samples the same Boltzmann
        distribution as sample(), but near the critical point, where single-spin Metropolis needs ~L^2 sweeps to
        forget its start, a few dozen cluster sweeps are enough; and every sample is independent. Needs J > 0."""
        if self.J <= 0:
            raise ValueError("Swendsen-Wang needs a ferromagnetic coupling J > 0.")
        L = self.L
        spins = self.rng.choice([-1, 1], size=(n_samples, L, L))
        site = np.arange(spins.size).reshape(spins.shape)
        neighbours = [np.roll(site, -1, axis=1), np.roll(site, -1, axis=2)]  # below and to the right, periodic
        p_bond = 1.0 - np.exp(-2.0 * self.beta * self.J)
        for _ in range(sweeps):
            flat = spins.ravel()
            heads, tails = [], []
            for other in neighbours:
                bond = (flat[site] == flat[other]) & (self.rng.random(spins.shape) < p_bond)
                heads.append(site[bond])
                tails.append(other[bond])
            head, tail = np.concatenate(heads), np.concatenate(tails)
            graph = coo_matrix((np.ones(len(head)), (head, tail)), shape=(spins.size, spins.size))
            n_clusters, cluster = connected_components(graph, directed=False)
            flip = np.where(self.rng.random(n_clusters) < 0.5, -1, 1)
            spins = spins * flip[cluster].reshape(spins.shape)
        return Field(
            values=spins,
            scale=1.0,
            batched=True,
            metadata={"system": "2d_ising", "beta": self.beta, "J": self.J},
        )
