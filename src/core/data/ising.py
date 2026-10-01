from dataclasses import dataclass
import numpy as np

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