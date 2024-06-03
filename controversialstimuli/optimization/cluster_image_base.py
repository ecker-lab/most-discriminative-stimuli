from abc import ABC, abstractmethod
from copy import deepcopy
from functools import partial
import torch
from torchvision.transforms.functional import affine
from ..utility.model_utils import translate_model


translate = partial(affine, angle=0, scale=1, shear=0)


class ActMaxClustImgBase(ABC):
    center_locations = None
    canvas_size = None

    @abstractmethod
    def get_pred_loss(self) -> torch.Tensor:
        """Returns the loss for the model predictions."""
        pass

    @abstractmethod
    def get_reg_loss(self) -> torch.Tensor:
        """Returns the regularization loss on the optimized stimuli."""
        pass

    def init_model(self, model):
        model = model.to("cpu")
        model = deepcopy(model)
        model = model.to(self.device).eval()
        model_shifted_readout = translate_model(
            model, self.center_locations, self.canvas_size
        )
        return model_shifted_readout

    @staticmethod
    def max_pred_ac_rot(predictions: torch.Tensor) -> torch.Tensor:
        """Returns the maximum prediction across orientations."""
        if predictions.dim() == 2:
            predictions = torch.max(predictions, dim=0)[
                0
            ]  # max of predictions across orientations
        elif predictions.dim() == 3:  # (batch, rot, units)
            predictions = torch.max(predictions, dim=1)[0]
        elif predictions.dim() == 0:
            pass
        else:
            raise ValueError("Invalid dimension.")
        return predictions

    def get_loss(self) -> torch.Tensor:
        """Returns the total loss."""
        return self.get_pred_loss() + self.get_reg_loss()
