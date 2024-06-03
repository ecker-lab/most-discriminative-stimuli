from copy import deepcopy
from itertools import combinations
from typing import Dict, List, Optional, Tuple
import logging
import random
import numpy as np
from tqdm.auto import tqdm

from ..optimization.torch_transform_image import ActMaxTorchTransfBaseUni
from ..optimization.controversial_objectives import ContrastiveAcrossImgs, ContrastiveNeuron, ContrastiveNeuronUnif
from .kmeans import KMeans


LOGGER = logging.Logger(__name__)


class SplitMerge:
    def __init__(
        self,
        img_optimizer: ActMaxTorchTransfBaseUni,
        unit_idc: List[int],  # MUST MATCH MODEL OUTPUT
        cluster_idc: List[int],  # ALL CLUSTER INDICES, e.g. list(range(num_clusters))
        init_cluster_assignments: Optional[Tuple[Dict[int, List[int]]]] = None,  # maps cluster index to neuron index
        seed: int = 42,
        max_iter_global: Optional[int] = None,
        max_iter_split_kmeans: Optional[int] = None,
        # sim_metric: str = "correlation",  # TODO later
        objective_improvement_threshold: float = 0.1,
        verbose=False,
        device="cuda",
    ):
        """_summary_

        Args:
            unit_idc (List): indices_of_all_units
            max_iter (Optional[int], optional): If None, max_iter is not evaluated. Defaults to None.
            num_clusters // clust_assignments: provide either one or the other; init_num_clusters will initialize clusters randomly
        """
        self._img_optimizer = img_optimizer
        self._unit_idc = unit_idc
        self._cluster_idc = cluster_idc
        self._init_cluster_assignments = init_cluster_assignments
        self._seed = seed
        self._verbose = verbose
        self._device = device
        self._max_iter_global = max_iter_global
        self._max_iter_split_kmeans = max_iter_split_kmeans
        # self._sim_metric = sim_metric
        self._objective_improvement_threshold = objective_improvement_threshold

    def run(self):
        clust_assign_changed = True
        while clust_assign_changed:
            global_kmeans = KMeans(
                img_optimizer=self._img_optimizer,
                unit_idc=self._unit_idc,
                cluster_idc=self._cluster_idc,
                optim_cluster_subset_idc=None,
                init_cluster_assignments=self._init_cluster_assignments,
                seed=self._seed,
                max_iter=self._max_iter_global,
                verbose=self._verbose,
                device=self._device,
            )
            clust_assign_changed = global_kmeans.run()
            self._cluster_img_dict = global_kmeans._cluster_img_dict
            self._cluster_assignments = global_kmeans._cluster_assignments  # dict, cluster_id -> neuron_id
            self._neuron_id2cluster_dict = global_kmeans._neuron_id2cluster_dict
            global_predictions = (
                global_kmeans._predict()
            )  # (self._clust_idc, self._unit_idc) shaped ndarray of model responses to the cluster images

            split_clust_assign_changed, modified_cluster_idc = self._split(global_predictions)
            clust_assign_changed = split_clust_assign_changed or clust_assign_changed
            clust_assign_changed = (
                self._merge(global_predictions, ignore_clust_idc=modified_cluster_idc) or clust_assign_changed
            )

        self._save_results()

    def _split(self, predictions: np.array) -> bool:
        """
        Returns:
            bool: True if cluster assignments changed, False otherwise
        """
        # find candidates for splitting
        if not isinstance(self._img_optimizer.objective_fct, ContrastiveNeuronUnif):
            raise NotImplementedError("Verify if implemented for other objective functions")

        clust_assign_changed = False
        new_cluster_idc = []
        modified_cluster_idc = []
        new_cluster_img_dict = {}
        new_cluster_assignments = {}
        new_neuron_id2cluster_dict = {}
        for clust_idx, pred in tqdm(zip(self._cluster_idc, predictions), leave=False):
            self._img_optimizer.objective_fct.set_clusters(
                on_clust_idx=clust_idx,
                off_clust_idc=list(set(self._cluster_idc) - set([clust_idx])),
                clust_assignments=np.array([self._neuron_id2cluster_dict[n] for n in sorted(self._unit_idc)]),
            )
            init_loss = -self._img_optimizer.objective_fct(pred.unsqueeze(0)).cpu().numpy()

            # get neurons
            units_new_1 = set(
                random.Random(self._seed).sample(
                    self._cluster_assignments[clust_idx], len(self._cluster_assignments[clust_idx]) // 2
                )
            )
            units_new_2 = sorted(list(set(self._cluster_assignments[clust_idx]) - units_new_1))
            units_new_1 = sorted(list(units_new_1))

            # create new cluster
            new_clust_idx = max(self._cluster_idc + new_cluster_idc) + 1

            # update cluster assignments
            cluster_assignments = deepcopy(self._cluster_assignments)
            cluster_assignments[clust_idx] = units_new_1
            cluster_assignments[new_clust_idx] = units_new_2

            kmeans = KMeans(
                img_optimizer=self._img_optimizer,
                unit_idc=self._unit_idc,
                cluster_idc=self._cluster_idc + [new_clust_idx],
                optim_cluster_subset_idc=(clust_idx, new_clust_idx),
                init_cluster_assignments=cluster_assignments,
                seed=self._seed,
                max_iter=self._max_iter_split_kmeans,
                verbose=self._verbose,
                device=self._device,
            )
            kmeans.run()

            loss_lst = []
            for i, pred in zip((clust_idx, new_clust_idx), kmeans._predict()):
                self._img_optimizer.objective_fct.set_clusters(
                    on_clust_idx=clust_idx,
                    off_clust_idc=list(set(self._cluster_idc) - set([clust_idx])),
                    clust_assignments=np.array([self._neuron_id2cluster_dict[n] for n in sorted(self._unit_idc)]),
                )
                loss_lst.append(-self._img_optimizer.objective_fct(pred.unsqueeze(0)).cpu().numpy())
            new_loss = np.mean(loss_lst)

            if new_loss < init_loss * (1 - self._objective_improvement_threshold):
                new_cluster_assignments.update(kmeans._cluster_assignments)
                new_neuron_id2cluster_dict.update(kmeans._neuron_id2cluster_dict)
                new_cluster_img_dict.update(kmeans._cluster_img_dict)
                new_cluster_idc.append(new_clust_idx)
                clust_assign_changed = True
                modified_cluster_idc.extend((clust_idx, new_clust_idx))

        # TODO think if this could result in a bug with we do the merging after that
        self._cluster_idc.append(new_cluster_idc)
        self._cluster_assignments.update(new_cluster_assignments)
        self._neuron_id2cluster_dict.update(new_neuron_id2cluster_dict)
        self._cluster_img_dict.update(new_cluster_img_dict)
        return clust_assign_changed, modified_cluster_idc

    def _merge(self, predictions: np.array, ignore_clust_idc) -> bool:
        """
        Returns:
            bool: True if cluster assignments changed, False otherwise
        """

        for clust1_idx, clust2_idx in tqdm(
            combinations(list(set(self._cluster_idc) - set(ignore_clust_idc))), leave=False
        ):
            loss_lst = []
            for i in (clust1_idx, clust2_idx):
                self._img_optimizer.objective_fct.set_clusters(
                    on_clust_idx=i,
                    off_clust_idc=list(set(self._cluster_idc) - set([i])),
                    clust_assignments=np.array([self._neuron_id2cluster_dict[n] for n in sorted(self._unit_idc)]),
                )
                loss_lst.append(-self._img_optimizer.objective_fct(predictions[i].unsqueeze(0)).cpu().numpy())
            init_loss = np.mean(loss_lst)

            # TODO merging, get new cluster idx

            # TODO optimize again without kmeans (or run kmeans with just that one cluster to consider, should work too and have same interface)

            pred = predictions[new_clust_idx]
            self._img_optimizer.objective_fct.set_clusters(
                on_clust_idx=new_clust_idx,
                off_clust_idc=list(set(self._cluster_idc) - set([new_clust_idx])),
                clust_assignments=np.array([self._neuron_id2cluster_dict[n] for n in sorted(self._unit_idc)]),
            )
            new_loss = -self._img_optimizer.objective_fct(new_pred.unsqueeze(0)).cpu().numpy()

            if new_loss < init_loss * (1 - self._objective_improvement_threshold):
                pass  # TODO)

    def _save_results(self, folder_path: str = "./split_merge_results"):
        pass

    def _get_mean_pred(self, predictions: np.array) -> np.array:
        mean_pred = [
            predictions[:, self._cluster_assignments[i]] for i in sorted(self._cluster_assignments.keys())
        ]  # (num_clusters, num_clusters), first index: images, last index: neurons
        return mean_pred
