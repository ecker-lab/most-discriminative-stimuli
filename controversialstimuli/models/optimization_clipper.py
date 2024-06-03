import torch


# based on https://github.com/sinzlab/mei
# - mei.legacy.utils.varargin
# - mei.legacy.ops.ChangeNormJointlyClipRangeSeparately
 
# originally published under the 

# MIT License

# Copyright (c) 2019 Cajal MICrONS Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


def varargin(f):
    """ Decorator to make a function able to ignore named parameters not declared in its
     definition.

    Arguments:
        f (function): Original function.

    Usage:
            @varargin
            def my_f(x):
                # ...
        is equivalent to
            def my_f(x, **kwargs):
                #...
        Using the decorator is recommended because it makes it explicit that my_f won't
        use arguments received in the kwargs dictionary.
    """
    import inspect
    import functools

    # Find the name of parameters expected by f
    f_params = inspect.signature(f).parameters.values()
    param_names = [p.name for p in f_params]  # name of parameters expected by f
    receives_kwargs = any(
        [p.kind == inspect.Parameter.VAR_KEYWORD for p in f_params]
    )  # f receives a dictionary of **kwargs

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        if not receives_kwargs:
            # Ignore named parameters not expected by f
            kwargs = {k: kwargs[k] for k in kwargs.keys() if k in param_names}
        return f(*args, **kwargs)

    return wrapper


class OptimizationClipperNoop:
    """ No-op clipper. """

    @varargin
    def __call__(self, x):
        return x


class ChangeNormJointlyClipRangeSeparately:
    """ Change the norm, then clip the value of x to some specified range
    Arguments:
        norm (float):   Desired norm
        x_min (float):  Lower valid value
        x_max (float):  Higher valid value
    """

    def __init__(self, norm, x_min_green, x_max_green, x_min_uv, x_max_uv, device="cuda"):
        self.norm = norm
        self.x_min_green = x_min_green
        self.x_max_green = x_max_green
        self.x_min_uv = x_min_uv
        self.x_max_uv = x_max_uv
        self._device = device

    @varargin
    def __call__(self, x):
        """
        x: torch tensor of shape 1 x channels x time x n_rows x n_cols
        """
        x_norm = torch.norm(x.view(len(x), -1), dim=-1)
        renorm = x * (self.norm / x_norm).view(len(x), *[1] * (x.dim() - 1))

        # create torch tensors with the min and max values for uv and green
        x_min_green = torch.full(size=x[:, 0].shape, fill_value=self.x_min_green)
        x_max_green = torch.full(size=x[:, 0].shape, fill_value=self.x_max_green)
        x_min_uv = torch.full(size=x[:, 0].shape, fill_value=self.x_min_uv)
        x_max_uv = torch.full(size=x[:, 0].shape, fill_value=self.x_max_uv)
        dichrom_min = torch.stack([x_min_green, x_min_uv], dim=1)
        dichrom_max = torch.stack([x_max_green, x_max_uv], dim=1)
        dichrom_max = dichrom_max.to(self._device)
        dichrom_min = dichrom_min.to(self._device)
        renorm_clipped = torch.max(
            torch.min(
                renorm, dichrom_max
            ), dichrom_min
        )
        return renorm_clipped
