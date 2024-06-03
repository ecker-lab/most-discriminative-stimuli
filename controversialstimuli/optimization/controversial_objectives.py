import logging
import warnings
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from controversialstimuli.utility.misc_helpers import CLIP_DIV_PRED_MIN

from ..utility.torch_nn import SimpleLogSoftmaxTemperature

EPSILON = 1e-6


logger = logging.getLogger(__name__)


class ObjectiveABC(ABC, nn.Module):
    def __init__(self, subt_pred: Optional[torch.Tensor]=None, div_pred: Optional[torch.Tensor]=None, clip_div_pred_min: float=CLIP_DIV_PRED_MIN):
        """Base class for objectives

        Args:
            subt_pred (Optional[torch.Tensor], optional): Subtract from predictions/logit before feeding into the 
                objective. Defaults to None.
            div_pred (Optional[torch.Tensor], optional): Divide predictions/logit before feeding into the objective.
                Defaults to None.
            clip_div_pred_min (float, optional): Minimum value for the division. Defaults to CLIP_DIV_PRED_MIN.
        """
        super().__init__()
        self.register_buffer(
            "_div_pred",
            torch.tensor(np.clip(div_pred, a_min=clip_div_pred_min, a_max=None))
            if div_pred is not None
            else torch.tensor(1),
        )
        self.register_buffer("_subt_pred", torch.tensor(subt_pred) if subt_pred is not None else torch.tensor(0))

    def _norm(self, pred: torch.Tensor) -> torch.Tensor:
            """
            Normalize the predicted tensor.

            Args:
                pred (torch.Tensor): The predicted tensor.

            Returns:
                torch.Tensor: The normalized tensor.
            """
            ret = (pred - self._subt_pred) / (self._div_pred + EPSILON)
            logger.debug(f"norm_pred mean: {ret.detach().cpu().numpy().mean()}")
            logger.debug(f"norm_pred std, {ret.detach().cpu().numpy().std()}")
            return ret

    def _check_clust_assignments(self, clust_assignments):
        """
        Check the validity of cluster assignments.

        Args:
            clust_assignments (list): List of cluster assignments.

        Raises:
            ValueError: If the length of `clust_assignments` does not match the length of `self._div_pred` or `self._subt_pred`.

        """
        if len(self._div_pred.shape) > 0 and len(clust_assignments) != len(self._div_pred):
            raise ValueError(
                f"clust_assignments and div_pred must have the same length. Got {len(clust_assignments)} and {len(self._div_pred)}"
            )
        if len(self._subt_pred.shape) > 0 and len(clust_assignments) != len(self._subt_pred):
            raise ValueError(
                f"clust_assignments and subt_pred must have the same length. Got {len(clust_assignments)} and {len(self._subt_pred)}"
            )

    @abstractmethod
    def forward(self, unit_scores: torch.FloatTensor) -> torch.FloatTensor:
        """Calculates the objective.
        Args:
            unit_scores: shape (num_neurons,) (dim 1), logits or firing rates

        Returns:
            torch.Tensor: float objective
        """
        pass

    @abstractmethod
    def set_clusters(
        self,
        on_clust_idx: int,
        off_clust_idc: Union[NDArray, List[NDArray]],
        clust_assignments: NDArray,
        unit_idc: Optional[List[int]],
    ):
        """(Re-)Set the clusters."""
        pass


class MeanNeuron(ObjectiveABC):
    def __init__(
        self,
        on_clust_idx: int,
        off_clust_idc: int,
        clust_assignments: NDArray,
        subt_pred=None,
        div_pred=None,
        clip_div_pred_min=CLIP_DIV_PRED_MIN,
    ):
        """supt_pred and div_pred always have to have the same shape/length as clust_assignments"""
        super().__init__(subt_pred, div_pred, clip_div_pred_min)
        self.set_clusters(on_clust_idx, off_clust_idc, clust_assignments)

    def set_clusters(
        self,
        on_clust_idx: int,
        off_clust_idc: Union[NDArray, List[NDArray]],
        clust_assignments: NDArray,
    ):
        self._check_clust_assignments(clust_assignments)
        assert (
            len(set(off_clust_idc).intersection({on_clust_idx})) == 0
        ), "on_clust_idx and off_clust_idc must be disjoint"
        self.register_buffer("_on_units_mask", torch.tensor(clust_assignments == on_clust_idx))

    def forward(self, unit_scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            unit_scores: shape (num_neurons,) (dim 1), logits or firing rates

        Returns:
            torch.Tensor: float
        """
        return self._norm(unit_scores)[self._on_units_mask].mean()


class ObjectiveIncrease(ObjectiveABC):
    def __init__(
        self,
        on_clust_idx: int = 0,
        off_clust_idc: Optional[NDArray] = None,
        clust_assignments: Optional[NDArray] = None,
    ):
        """Computes the maximally exciting stimulus for a single cluster.

        Args:
            on_clust_idx (int, optional): Index of cluster that should be driven. Defaults to 0.
            off_clust_idc (Optional[NDArray], optional): All other clusters. Defaults to None.
            clust_assignments (Optional[NDArray], optional): Cluster assignment indices. Defaults to None.
        """
        super().__init__()
        self._cluster_assignments: Optional[NDArray] = None
        self.set_clusters(on_clust_idx, off_clust_idc, clust_assignments)

    def forward(self, unit_scores: torch.FloatTensor) -> torch.FloatTensor:
        if self._cluster_assignments is not None:
            selected_unit_scores = unit_scores[0][self._on_units_mask]
        else:
            selected_unit_scores = unit_scores

        avg_firing_rate = selected_unit_scores.mean()
        assert avg_firing_rate.isfinite()
        return avg_firing_rate

    def set_clusters(
        self,
        on_clust_idx: int,
        off_clust_idc: Optional[Union[NDArray, List[NDArray]]],
        clust_assignments: Optional[NDArray],
        unit_idc: Optional[List[int]] = None,
    ):
        """Set the cluster assignments.

        Args:
            see __init__ function.
            unit_idc: Indices of the units to consider in the objective. Defaults to None, i.e. uses all.
        """
        if clust_assignments is not None:
            raise NotImplementedError("passing clust_assignments is not implemented.")
        if unit_idc is not None:
            raise NotImplementedError("passing unit_idc is not implemented.")
        
        self._cluster_assignments = clust_assignments
        self.register_buffer("_on_units_mask", torch.tensor(clust_assignments == on_clust_idx))


def contrastive_neuron_objective(
    on_logits_mean: torch.Tensor, logit_cluster_means: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    """Computes the contrastive objective Eq. (1) for a single on cluster.

    Args:
        on_logits_mean: The mean of the on cluster logits or predictions. Shape: (1,)
        logit_cluster_means: The mean of each clusters logits or predictions. Shape: (num_clusters,)
        temperature: The temperature of the softmax.

    Returns:
        The objective value. Shape: (1,)
    """
    t = temperature
    ll_means = logit_cluster_means
    obj = (1 / t) * on_logits_mean - torch.logsumexp((1 / t) * ll_means, dim=0) + torch.log(torch.tensor(len(ll_means)))
    return obj


class ContrastiveNeuronUnif(ObjectiveABC):
    def __init__(
        self,
        on_clust_idx: int,
        off_clust_idc: List[int],
        clust_assignments: List[int],
        temperature: float = 1,
        subt_pred: Optional[torch.Tensor] = None,
        div_pred: Optional[torch.Tensor] = None,
        clip_div_pred_min: float = CLIP_DIV_PRED_MIN,
        device: str = "cuda",
    ):
        """Computes the contrastive objective Eq. (1) for a single on cluster.
        
        Args:
            on_clust_idx: The index of the cluster that should be on.
            off_clust_idc: The indices of the clusters that should be off.
            clust_assignments: The cluster assignments of the neurons. Shape: (num_neurons,)
            temperature: The temperature of the softmax.
            subt_pred: see ObjectiveABC
            div_pred: see ObjectiveABC
            clip_div_pred_min: see ObjectiveABC
        """
        super().__init__(subt_pred=subt_pred, div_pred=div_pred, clip_div_pred_min=clip_div_pred_min)
        self._temperature = nn.Parameter(torch.tensor(temperature), requires_grad=False)
        self._device = device
        self._unit_idc = None
        self.set_clusters(on_clust_idx, off_clust_idc, clust_assignments)

    def set_clusters(self, on_clust_idx: int, off_clust_idc: Union[NDArray, List[NDArray]], clust_assignments: List[int], unit_idc: Optional[List[int]] = None):
        self._check_clust_assignments(clust_assignments)

        if isinstance(clust_assignments, torch.Tensor):
            clust_assignments = clust_assignments.tolist()
        elif isinstance(clust_assignments, np.ndarray):
            clust_assignments = list(clust_assignments)
        else:
            pass

        on_clust_idx = int(on_clust_idx)
        cluster_ids_without_on_cluster = sorted(set(clust_assignments) - {int(on_clust_idx)})
        on_off_cluster_set = {idx for idx in off_clust_idc}
        on_off_cluster_set.add(on_clust_idx)
        # on cluster will be at the first position
        cluster_ids_to_optimize = [idx for idx in [on_clust_idx] + cluster_ids_without_on_cluster if idx in on_off_cluster_set]

        cluster_masks_list = [torch.tensor(clust_assignments) == clust_idx for clust_idx in cluster_ids_to_optimize]
        cluster_masks = torch.stack(cluster_masks_list).to(self._device)
        assert torch.all(cluster_masks.sum(-1) > 0), "Some cluster did not have entries, this will lead to infinities"

        self._normalized_cluster_mask = cluster_masks / cluster_masks.sum(-1, keepdim=True)
        if unit_idc is not None:
            self._unit_idc = torch.tensor(unit_idc, requires_grad=False).to(self._device)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: The logits or predictions. Shape: (num_neurons,)

        Returns:
            The objective value. Shape: (1,)
        """
        assert logits.ndim == 2, "Backwards compatibility, could be generalized"
        assert logits.shape[0] == 1, "Backwards compatibility, could be generalized"
        logits = logits[0]
        if self._unit_idc is not None:
            logits = logits[self._unit_idc]
        logits = self._norm(logits)

        expanded_logits = torch.unsqueeze(logits, dim=0)
        l_clust_means = (expanded_logits * self._normalized_cluster_mask).sum(-1)
        # In set_clusters we made sure that the on_cluster is at the first position
        on_clust_mean = l_clust_means[0]
        loss = contrastive_neuron_objective(on_clust_mean, l_clust_means, self._temperature)

        assert loss.isfinite()
        return loss
