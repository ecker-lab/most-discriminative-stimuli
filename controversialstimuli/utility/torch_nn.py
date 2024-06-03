import warnings
from torch import Tensor, tensor, logsumexp
from torch.nn import Parameter, Module, LogSoftmax


class SimpleLogSoftmaxTemperature(Module):
    """A log softmax that can be used with a temperature parameter.
    The numerator is a single value, the denominator a 1D tensor.
    """

    def __init__(self, temperature: float = 1) -> None:
        """Initialize the SimpleLogSoftmax.
        
        Args:
            temperature: The temperature of the softmax.
        """
        super().__init__()
        self.temperature = Parameter(tensor(temperature), requires_grad=False)

    def forward(self, num: Tensor, denom: Tensor) -> Tensor:
        """Compute the log softmax.
        
        Args:
            num: The numerator of the log softmax. Shape: (1,)
            denom: The denominator of the log softmax. Shape: (N,)"""
        return num / self.temperature - logsumexp(denom / self.temperature, dim=0)
