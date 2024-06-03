import numpy as np
from torch import nn
import torch
from torch.nn import functional as F


def laplace_weights(size: int = 3) -> np.ndarray:
    if size == 3:
        return np.array(
            [[0.25, 0.5, 0.25], [0.5, -3.0, 0.5], [0.25, 0.5, 0.25]]
        ).astype(np.float32)[None, None, ...]

    else:
        raise ValueError("Only size=3 supported.")


class LaplaceFilter(nn.Module):
    def __init__(self, size: int = 3, padding: str = "reflect") -> None:
        """Initialize the laplace filter.

        Args:
            size (int): size of the filter.
            padding (str): padding mode. (str: valid, same, replicate)
        """

        super().__init__()
        self.register_buffer("weight", torch.from_numpy(laplace_weights(size=size)))

        if padding == "valid":
            self.pad = nn.Identity()

        elif padding == "same":
            self.pad = nn.ZeroPad2d(size - 1)

        elif padding == "replicate":
            self.pad = nn.ReplicationPad2d(size - 1)

        elif padding == "reflect":
            self.pad = nn.ReflectionPad2d(size - 1)

        else:
            raise ValueError(
                'Only padding="valid", "same", "replicate", "reflect" supported.'
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pad(x)
        return F.conv2d(x, self.weight)


class LaplaceLoss(nn.Module):
    def __init__(
        self, size: int = 3, padding: str = "reflect", mean: bool = True
    ) -> None:
        super().__init__()

        self.filter = LaplaceFilter(size, padding)
        self.agg_fn = torch.mean if mean else torch.sum

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        oc, ic, k1, k2 = x.shape
        return self.agg_fn(self.filter(x.view(oc * ic, 1, k1, k2)) ** 2)
