import logging
from functools import partial
from typing import Optional, List, Callable, Tuple

import datajoint as dj
import numpy as np
import torch
from torch import Tensor
import torchvision.transforms.functional as TF
from controversialstimuli.optimization.controversial_objectives import (
    ObjectiveABC,
)
from nnfabrik.utility.nn_helpers import set_random_seed

from ..regularizers import LaplaceLoss
from ..utility.misc_helpers import get_verbose_print_fct
from ..utility.layer_utils import sigmoid_centered
from .cluster_image_base import ActMaxClustImgBase
from .stoppers import EMALossStopper, ImgPxIntensityChangeStopper

EPSILON = 1e-6
logger = logging.getLogger(__name__)
translate = partial(TF.affine, angle=0, scale=1, shear=0)


class ActMaxTorchTransfBaseUni(ActMaxClustImgBase):
    def __init__(
        self,
        model: torch.nn.Module,
        canvas_size: Tuple[int, ...],  # (h, w) or (t, h, w)
        optimizer_init_fn: Callable[[List[torch.Tensor]], torch.optim.Optimizer],
        stopper: Optional[Callable],
        objective_fct: ObjectiveABC,
        smoothness: float = 0,
        max_iter: int = 5000,
        seed: int = None,
        device: str = "cuda",
        verbose: bool=False,
        dj_ping: bool=False,
        grad_precond_fct: Optional[Callable]=None,
        num_imgs: int=1,
        postprocessing: Optional[Callable] = None,
    ) -> None:
        """Stimulus optimizer class.

        Args:
            model (torch.nn.Module): Model predicting neural activity or model logits
            canvas_size (Tuple[int, ...]): (h, w) or (t, h, w) of model inputs
            stopper (Optional[Callable]): from .stoppers, criterion to stop optimization early
            objective_fct (ObjectiveABC): from .objectives, objective function to optimize
            smoothness (float, optional): Laplace smoothness loss regularization weight
            max_iter (int, optional): Maximum stimulus optimization iteration steps. Defaults to 5000.
            seed (int, optional): Defaults to None.
            device (str, optional): Defaults to "cuda".
            verbose (bool, optional): Verbose output. Defaults to False.
            dj_ping (bool, optional): Ping datajoint database to keep connection alive. Defaults to False.
            grad_precond_fct (Optional[Callable], optional): Function that is applied to the gradients on the stimulus 
                before the optimization step is executed. For example mei.legacy.ops.GaussianBlur. Defaults to None.
            num_imgs (int, optional): Humber of images to simultaneously optimize. Defaults to 1.
            postprocessing (Optional[Callable], optional): Function that is applied to the stimulus after the 
                optimization step. For example mei.legacy.ops.ChangeNorm. Defaults to None.
        """
        self.device = device

        self.seed = seed
        if self.seed is not None:
            set_random_seed(self.seed)

        self.canvas_size = canvas_size
        self.smoothness = smoothness
        self.stopper = stopper
        self.max_iter = max_iter
        self.verbose_print = get_verbose_print_fct(verbose)
        self.dj_ping = dj_ping

        self.grad_precond_fct = grad_precond_fct if grad_precond_fct is not None else torch.nn.Identity()
        self.num_imgs = num_imgs

        self._postprocessing = postprocessing
        self._img = self.init_img(canvas_size)  # (batch, chan=1, h, w)
        self.get_laplace_loss = LaplaceLoss().to(device)
        self._optimizer_init_fn = optimizer_init_fn
        self.optim = self._optimizer_init_fn([self._img])

        self.model = model.to(device).eval()

        if objective_fct is not None:
            self.objective_fct = objective_fct.to(device)
        else:
            self.objective_fct = None

    def reset(self, initial_stimulus: Optional[torch.FloatTensor]) -> None:
        """Reset the stimulus optimizer state.
        
        Args:
            initial_stimulus (Optional[torch.FloatTensor]): Initialize the reset stimulus with `initial_stimulus`. 
                Defaults to None (stimulus is randomly initialized).
        """
        self.verbose_print("Resetting image optimizer")
        if self.stopper is not None:
            self.stopper.reset()
        if initial_stimulus is None:
            self._img = self.init_img(self.canvas_size)
        else:
            self._img = initial_stimulus.detach().requires_grad_()
        self.optim = self._optimizer_init_fn([self._img])

    def init_img(self, canvas_size: Tuple[int, ...]) -> Tensor:
        """Randomly initialize the stimulus."""
        if len(canvas_size) >= 3:
            shape = (self.num_imgs, ) + canvas_size
        else:
            if canvas_size[0] != canvas_size[1]:
                img_h = np.ceil(np.sqrt(canvas_size[0] ** 2 + canvas_size[1] ** 2)).astype(int)
                img_w = img_h
            else:
                img_h = canvas_size[0]
                img_w = canvas_size[1]
            shape = (self.num_imgs, 1, img_h, img_w)
        img = torch.randn(shape, requires_grad=True, device=self.device)
        if self._postprocessing is not None:
            img.data = self._postprocessing(img.data)
        return img

    @property
    def img(self) -> Tensor:
        """Can be overwritten by derived classes such that image is returned after L2 normalization, sigmoid 
            constraint, etc.
        """
        img = self._img
        return img

    def get_reg_loss(self):
        """See parent class."""
        if self.smoothness == 0.0:
            return 0.0
        else:
            return self.smoothness * self.get_laplace_loss(self.img)

    def get_pred_loss(self) -> Tensor:
        """See parent class."""
        preds = self.model(self.img)  # (batch, units)
        return -self.objective_fct(preds)

    def optim_step(self) -> Tensor:
        """Perform one stimulus optimization step."""
        loss = self.get_loss()
        self.optim.zero_grad()
        loss.backward()
        self._img.grad = self.grad_precond_fct(self._img.grad)
        self.optim.step()
        if self._postprocessing is not None:
            self._img.data = self._postprocessing(self._img.data)
        return loss

    def maximize(self) -> None:
        """Maximize the objective function, performing the full optimization."""
        self.loss_lst = []

        for i in range(self.max_iter):
            if self.dj_ping and not i % 10:
                dj.conn().ping()

            loss_i = self.optim_step()

            self.loss_lst.append(loss_i.item())
            if not i % 10:
                self.verbose_print(
                    f"iter = {i:4d}, loss = {loss_i:.9f}, pred_loss = {self.get_pred_loss():.9f}, reg_loss = {self.get_reg_loss():.9f}"
                )

            if self.stopper is not None and isinstance(self.stopper, EMALossStopper):
                self.stopper.update(self.get_pred_loss().detach())
            elif self.stopper is not None and isinstance(self.stopper, ImgPxIntensityChangeStopper):
                self.stopper.update(self.img)

            if self.stopper is not None and self.stopper.criterion_hit:
                print("Stopper criterion reached")
                return self.loss_lst[-1]

        print(f'max_iter {self.max_iter} reached')
        return self.loss_lst[-1]

