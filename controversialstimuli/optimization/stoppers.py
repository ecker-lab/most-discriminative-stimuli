from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from controversialstimuli.utility.misc_helpers import get_verbose_print_fct


class EMALossStopper:
    """Exponential moving averaged loss stopper."""

    def __init__(self, patience: int = 100, ema_weight: float = 0.9, verbose: bool = False) -> None:
        self.min_loss = 1e10
        self.not_decreased = 0
        self.ema_weight = ema_weight
        self.patience = patience
        self.ema_loss = None
        self.__criterion_hit = False

        self.verbose_print = get_verbose_print_fct(verbose)

    def update(self, loss: float) -> None:
        if self.ema_loss is None:
            self.ema_loss = loss
        else:
            self.ema_loss = self.ema_weight * self.ema_loss + (1 - self.ema_weight) * loss

        if self.ema_loss < self.min_loss:
            self.min_loss = self.ema_loss
            self.not_decreased = 0
        else:
            self.not_decreased += 1
            self.verbose_print(f"Not decreased for {self.not_decreased} steps out of {self.patience} patience steps")

            if self.not_decreased > self.patience:
                self.__criterion_hit = True
                self.verbose_print("Stopper criterion hit")

    @property
    def criterion_hit(self) -> bool:
        return self.__criterion_hit
    
    def reset(self) -> None:
        self.min_loss = 1e10
        self.not_decreased = 0
        self.ema_loss = None
        self.__criterion_hit = False


class ImgPxIntensityChangeStopper:
    """Optimization stopper as used by Golan et al. 2020 monitoring the maximum change of pixel intensities.
    Stop if maximum change of pixel intensities is less than half of the screen's intensity level for `patience` steps."""

    def __init__(self, vmin: float, vmax: float, patience: int = 50, verbose: bool = False) -> None:
        """
        Args:
            vmin (float): Minimum intensity of input image. E.g. be -amplitude of centered sigmoid.
            vmax (float): Maximum intensity of input image. E.g. be amplitude of centered sigmoid.
        """

        if np.isclose(vmin, vmax):
            raise ValueError("vmin and vmax arguments must be different")

        self.steps_wo_change = 0
        self.patience = patience
        self.verbose_print = get_verbose_print_fct(verbose)
        self.vmin = vmin
        self.vmax = vmax
        self.monitor_max_intensity = 255.0
        self.prev_img = None
        self.__criterion_hit = False

    def update(self, img: torch.Tensor) -> None:
        if self.prev_img is None:
            self.prev_img = self.map_to_monitor_range(img).detach().clone()
            return

        curr_img = self.map_to_monitor_range(img)
        max_change = (curr_img - self.prev_img).abs().max().item()
        self.verbose_print("max. abs. img. change", np.round(max_change, 2), "monitor intensity levels")
        self.prev_img = curr_img.detach().clone()

        if max_change >= 0.5:  # half screen intensity level
            self.steps_wo_change = 0
        else:
            self.steps_wo_change += 1
            self.verbose_print(
                f"Img. intensity not changed more than half screen intensity level for {self.steps_wo_change} steps out of {self.patience} patience steps"
            )

            if self.steps_wo_change > self.patience:
                self.__criterion_hit = True
                self.verbose_print("Stopper criterion hit")

    def map_to_monitor_range(self, img: torch.Tensor) -> torch.Tensor:
        ret = (img - self.vmin) / (self.vmax - self.vmin) * self.monitor_max_intensity
        return ret

    @property
    def criterion_hit(self) -> bool:
        return self.__criterion_hit
    
    def reset(self) -> None:
        self.steps_wo_change = 0
        self.prev_img = None
        self.__criterion_hit = False


