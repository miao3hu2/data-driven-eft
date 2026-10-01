from dataclasses import dataclass
import numpy as np

from core.coarse_graining.base import CoarseGrainer
from core.data.base import Field

@dataclass
class MajorityBlockSpin(CoarseGrainer):
    block_size: int = 2
    seed: int | None = None

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def transform(
        self,
        field: Field,
    ) -> Field:
        values = field.values
        n_lead = field.n_lead
        n_space = field.n_space
        if any(d % self.block_size for d in field.spatial_shape):
            raise ValueError(
                f"spatial_shape {field.spatial_shape} is not divisible by block_size {self.block_size}"
            )
        new_spatial_dims = [d // self.block_size for d in field.spatial_shape]

        # Split every spatial axis n into (n // b, b): (*lead, n1, b, n2, b, ..., *components)
        blocked_shape = list(field.lead_shape)
        for d in new_spatial_dims:
            blocked_shape += [d, self.block_size]
        blocked_shape += list(field.component_shape)

        reshaped_values = values.reshape(blocked_shape)
        block_axes = tuple(n_lead + 2 * i + 1 for i in range(n_space))
        new_values = np.sign(np.sum(reshaped_values, axis=block_axes)) # A majority rule for block spins
        # Break ties (zero-sum blocks) randomly so the majority rule doesn't bias the magnetization
        ties = new_values == 0
        new_values[ties] = self.rng.choice([-1, 1], size=int(ties.sum()))

        new_scale = tuple(s * self.block_size for s in field.scale)
        return field.replace(values=new_values, scale=new_scale)
