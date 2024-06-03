"""Simple analysis connected to models and its predictions."""

import warnings
from typing import Dict, Iterable, Optional, Union

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import CenterCrop

from ..utility.misc_helpers import CLIP_DIV_PRED_MIN


def norm_unit_act_imgs(
    preds: NDArray,
    type_: Optional[str] = None,
    rep_dataloader: Optional[DataLoader] = None,
    stds: Optional[NDArray] = None,
    means: Optional[NDArray] = None,
) -> NDArray:
    """Normalize activity of each unit across image dimension (0-th dim)

    Args:
        preds (NDArray): shape (num_images, num_units)
        type (Optional[str], optional): Normalization type, one of 'std', 'z-score'., 'z-score-clip-ext': ust stds and
            means argument for normalization. Defaults to None.

    Returns:
        NDArray: Same shape as `preds`, but each unit normalized.
    """
    if type_ is None:
        pass

    elif type_ == "max" and rep_dataloader is None and stds is None and means is None:
        preds = preds / preds.max(0, keepdims=True)

    elif (
        type_ == "std-clip"
        and rep_dataloader is None
        and stds is None
        and means is None
    ):
        preds = preds / np.clip(
            np.std(preds, axis=0, keepdims=True, ddof=1),
            a_min=CLIP_DIV_PRED_MIN,
            a_max=None,
        )

    elif (
        type_ == "z-score-clip"
        and rep_dataloader is None
        and stds is None
        and means is None
    ):
        preds = (preds - np.mean(preds, axis=0, keepdims=True)) / np.clip(
            np.std(preds, axis=0, keepdims=True, ddof=1),
            a_min=CLIP_DIV_PRED_MIN,
            a_max=None,
        )
        print(
            f"clipping {np.sum(np.std(preds, axis=0, keepdims=True, ddof=1) < CLIP_DIV_PRED_MIN)} stds to {CLIP_DIV_PRED_MIN}"
        )

    elif (
        type_ == "z-score-clip-ext"
        and rep_dataloader is None
        and stds is not None
        and means is not None
    ):
        print("preds.shape", preds.shape)
        preds = (preds - means) / np.clip(stds, a_min=CLIP_DIV_PRED_MIN, a_max=None)
        print(
            f"clipping {np.sum(stds < CLIP_DIV_PRED_MIN)} stds to {CLIP_DIV_PRED_MIN}"
        )

    elif (
        type_ == "z-score-clip-coeff-of-var-ext"
        and rep_dataloader is None
        and stds is not None
        and means is not None
    ):
        denom = stds.copy()
        denom[stds < 0.01 * means] = 0.01 * means[stds < 0.01 * means]
        preds = (preds - means) / denom
        print(
            f"clipping {np.sum(stds < 0.01 * means)} stds to 0.01 * mean {0.01 * means[stds < 0.01 * means]}"
        )

    elif (
        type_ == "rep-mean_max"
        and rep_dataloader is not None
        and stds is not None
        and means is not None
    ):
        if next(iter(rep_dataloader)).responses.shape[0] != 10:
            warnings.warn("Check if repeats dataloader passed")
        mean_resps = np.stack(
            [batch.responses.mean(0).detach().cpu().numpy() for batch in rep_dataloader]
        )
        max_resp = mean_resps.max(0)
        preds /= max_resp

    else:
        raise ValueError(
            "Invalid comibation of `type_`, `rep_dataloader`, `stds` and `means` arguments passed."
        )

    return preds


def pred_max_rot(transl_model, canvas_size, imgs, device="cuda"):
    """Get maximum prediction across rotations.

    Args:
        transl_model (): model with readout maps shifted to be centered for all units
        canvas_size ():
        clust_imgs (): cluster images to compute confusion matrix for. Shape (n_clusters=1, n_rotations, h, w)
        device (str):

    Returns:
        NDArray: (n_images, n_neurons)
    """
    crop_to_canvas = CenterCrop(size=canvas_size)
    preds = []
    for img in imgs:
        # preds.append(model(torch.tensor(cMEIs).to(device)).detach().cpu().numpy())
        preds.append(
            transl_model(crop_to_canvas(torch.tensor(img).squeeze(0).to(device)))
            .detach()
            .cpu()
            .numpy()
        )

    preds = np.stack(preds, axis=0)  # (n_clusters, n_rotations, n_neurons)
    preds = np.max(preds, axis=1)  # (n_clusters, n_neurons)
    return preds
