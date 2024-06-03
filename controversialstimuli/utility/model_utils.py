import logging
import warnings
from functools import partial
from typing import Dict, Optional

from git import Union
import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn
from torchvision.transforms.functional import affine

from neuralpredictors.layers.readouts import FullGaussian2d, FullFactorized2d

translate = partial(affine, angle=0, scale=1, shear=0)


def get_input_shape(dataloaders: Dict[str, torch.utils.data.DataLoader]) -> tuple:
    """Returns the shape of input images in the dataloader, i.e. `torch.Size([chan, h, w])."""
    dataloader = next(iter(dataloaders["train"].values()))
    input_shape = dataloader.dataset[0].images.shape
    return input_shape


def load_transl_model(model_tab, key, rf_loc):
    """_summary_

    Args:
        model_tab (_type_): _description_
        key (_type_): _description_
        rf_loc (NDArray or List): (Num neurons, coordinates=2) where coordinates are (x and y) == (w and h) coordinates

    Returns:
        _type_: _description_
    """
    dataloaders, model = model_tab().load_model(key)
    model.to("cuda")
    model.eval()

    canvas_size = get_input_shape(dataloaders)[1:]
    model = translate_model(model, rf_loc, canvas_size)
    return dataloaders, model


def translate_model(model: nn.Module, rf_loc: Optional[Union[NDArray, list]]=None, canvas_size: Optional[tuple[int, int]]=None) -> nn.Module:
    """Translates model readout receptive field positions. Works for ensembles, Gaussian readout and factorized readout.

    Args:
        model (nn.Module): the model in its typical neuralpredictors core and readout structure.
        rf_loc (NDArray or List): (Num neurons, coordinates=2) where coordinates are (x and y) == (w and h) coordinates. If none, all readout positions will be shifted to the canvas center.
        canvas_size (tuple[int, int]): The shape of the model's input images as height, width.

    Returns:
        nn.Module: model with all readout positions shifted to the specified locatins
    """

    def translate_member_model(member, rf_loc, canvas_size):
        single_readout = next(iter(member.readout.values()))

        if isinstance(single_readout, FullGaussian2d):
            if rf_loc is not None:
                center_loc_flipped = np.flip(rf_loc, 1)  # height, width
                center_loc_rel = canvas_coord_abs2rel(center_loc_flipped)
                return transl_gauss_readout_pos(member, rf_loc=center_loc_rel)
            return transl_gauss_readout_pos(member, rf_loc=None)

        else:
            raise NotImplementedError(f"`translate_model` cannot handle `{single_readout}` readout type")

    if hasattr(model, "members"):
        for member in model.members:
            member = translate_member_model(member, rf_loc, canvas_size)
    else:
        model = translate_member_model(model, rf_loc, canvas_size)

    return model



def transl_gauss_readout_pos(model, rf_loc=None):
    """For models using the Gaussian readout neuralpredictors.layers.readouts.FullGaussian2d.
        Recommended to call translate_model instead.

    Args:
        model (torch model): _description_
        rf_loc (NDArray): for each unit, location of the RF center. If passed, locations will be shifted according
            to this value. Otherwise, all readout locations will be shifted to the canvas center. Shape: (num_units, 2)
            where "2" are hight, width / x, y coordinates relative to the canvas size / feature maps, in the -1, 1 interval

    Raises:
        NotImplementedError: _description_

    Returns:
        torch model: _description_
    """
    if len(model.readout.values()) > 1:
        raise NotImplementedError("Handling for different readout data_keys not implemented")

    training_status = model.training
    model.eval()
    readout = next(iter(model.readout.values()))

    with torch.no_grad():
        if rf_loc is None:
            logging.info("No rf locs were passed. Centering based on gaussian readout.")
            readout._mu.zero_()
            assert ~torch.any(readout.mu)
            assert ~torch.any(readout.sample_grid(1))
        else:
            # mu_shape = readout._mu.shape
            print("RF_LOC", rf_loc)
            readout._mu -= torch.tensor(rf_loc).view(readout._mu.shape).to(readout._mu.device)

    model.train(training_status)

    return model


def canvas_coord_abs2rel(locations: NDArray, canvas_size: tuple[int, int]=(36, 64)) -> NDArray:
    """Transforms absolute canvas coordinates to relative ones in the -1, 1 interval.

    Args:
        locations (NDArray): (Height, width)
        canvas_size (tuple, optional): _description_. Defaults to (36, 64).

    Returns:
        _type_: (Height, width)
    """
    return ((locations / np.array(canvas_size)) - 0.5) * 2


