import logging
import os
from typing import Any, List, Optional, Tuple
from abc import ABC, abstractmethod

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from torchvision.transforms import CenterCrop

from controversialstimuli.utility.pixel_image import RotateImg


logger = logging.getLogger(__name__)


class ClusteringModelABC(ABC, torch.nn.Module):
    def __init__(
        self,
        centered_ensemble_model: torch.nn.Module,
        return_logits: bool = False,
        neuron_list: Optional[List[int]] = None,
    ):
        """Base class for neural predictive models defining the interface for the clustering algorithm.

        Args:
            centered_ensemble_model (nn.Module): Ensemble of the predictive model with readout receptive field 
                positions being centered.
            return_logits (bool, optional): Whether to return logits of the model or neural activity. 
                Defaults to False.
            neuron_list (Optional[List[int]], optional): List of neurons that should be returned. Defaults to None (all
                returned).
        """
        super().__init__()
        self._ensemble_model = centered_ensemble_model
        self._return_logits = return_logits
        self._neuron_list = neuron_list

    def update_neuron_list(self, neuron_list: Optional[List[int]]) -> None:
        self._neuron_list = neuron_list

    @abstractmethod
    def forward(self, inputs: torch.Tensor, kwargs: dict[str, Any]) -> torch.Tensor:
        """return logits of each neuron for its best rotation (if rotation is required)
            needs to work with ensemble models
            should also handle normalization
            and also constrain the output to neuron_list, returning one vector of shape (batch, (optionaly time?), num_neurons)
        """
        pass


class TestOnOffModel(ClusteringModelABC):
    def __init__(self, number_of_neurons: int):
        neuron_list = list(range(number_of_neurons))
        super().__init__(
            centered_ensemble_model=torch.nn.Module(),  # type: ignore
            return_logits=False,
            neuron_list=neuron_list,
        )  # type: ignore

    def forward(self, inputs: torch.Tensor, kwargs=None) -> torch.Tensor:
        """ Returns the mean of the stimulus if the neuron index is even, otherwise returns the negative mean."""

        # take the mean over all dimension except the first batch dimension
        stimulus_mean = inputs.mean(dim=tuple(range(1, len(inputs.shape))))

        list_of_responses = []
        for neuron_index in self._neuron_list:
            if neuron_index % 2 == 0:
                response = stimulus_mean
            else:
                response = -stimulus_mean
            list_of_responses.append(response)

        stacked_responses = torch.stack(list_of_responses, dim=-1)
        return stacked_responses
