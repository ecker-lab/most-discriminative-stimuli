from copy import deepcopy
import os
import pickle
from typing import Tuple

import torch
import yaml

import mei.modules
import nnfabrik.builder


class Center:
    """
    Class centering readouts
    """

    def __init__(self, target_mean, mean_key="mask_mean"):
        self.target_mean = target_mean
        self.mean_key = mean_key

    def __call__(self, model):
        key_components = [
            comp
            for comp in zip(
                *[
                    key.split(".")
                    for key in model.state_dict().keys()
                    if self.mean_key in key
                ]
            )
        ]
        mean_full_keys = [
            ".".join([key_components[j][i] for j in range(len(key_components))])
            for i, var_name in enumerate(key_components[-1])
        ]

        mod_state_dict = deepcopy(model.state_dict())
        device = mod_state_dict[mean_full_keys[0]].device
        for mean_full_key in mean_full_keys:
            mod_state_dict[mean_full_key] = torch.zeros_like(
                model.state_dict()[mean_full_key]
            ) + torch.tensor(self.target_mean, device=device)
        model.load_state_dict(mod_state_dict)


def load_ensemble_retina_model_from_directory(
    directory_path: str, device: str = "cuda"
) -> Tuple:
    """
    Returns an ensemble data_info object and an ensemble model that it loads from the directory path.

    The code assumes that the following files are in the directory:
    - state_dict_{seed:05d}.pth.tar
    - config_{seed:05d}.yaml
    - data_info_{seed:05d}.pkl
    where seed is an integer that represents the random seed the model was trained with
    """

    file_names = [f for f in os.listdir(directory_path) if f.endswith("yaml")]
    seed_array = [
        int(file_name[: -len(".yaml")].split("_")[1]) for file_name in file_names
    ]
    model_list = []
    data_info_list = []

    for seed in seed_array:
        state_dir_path = f"{directory_path}/state_dict_{seed:05d}.pth.tar"
        model_config_path = f"{directory_path}/config_{seed:05d}.yaml"
        data_info_path = f"{directory_path}/data_info_{seed:05d}.pkl"

        state_dict = torch.load(state_dir_path)
        with open(model_config_path, "r") as f:
            config = yaml.safe_load(f)

        with open(data_info_path, "rb") as f:
            data_info = pickle.load(f)
        data_info_list.append(data_info)

        model_fn = config["model_fn"]
        model_config = config["model_config"]
        model = nnfabrik.builder.get_model(
            model_fn,
            model_config,
            seed=seed,
            data_info=data_info,
            state_dict=state_dict,
        )
        model_list.append(model)

    ensemble_model = mei.modules.EnsembleModel(*model_list)
    model_transform = Center(target_mean=[0.0, 0.0])
    model_transform(ensemble_model)
    ensemble_model.to(device)
    ensemble_model.eval()

    # Based on the mei mixin code normally the first data_info entry is used, they should be all the same
    # See: https://github.com/eulerlab/mei/blob/d1f24ef89bdeb4643057ead2ee6b1fb651a90ebb/mei/mixins.py#L88-L94
    data_info = data_info_list[0]

    return data_info, ensemble_model
