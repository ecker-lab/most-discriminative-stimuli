from abc import ABC, abstractmethod
import copy
import datetime
from math import ceil
import os
import pickle
from typing import Dict, List, Optional, Tuple, Set, Any, Union
from collections import defaultdict, Counter
import logging
import random
import time
import numpy as np
import torch
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix, adjusted_rand_score
import matplotlib.pyplot as plt
import seaborn

from controversialstimuli.utility.misc_helpers import get_verbose_print_fct, pickle_dump
from controversialstimuli.utility.retina.constants import RGC_GROUP_NAMES_DICT

from ..optimization.torch_transform_image import ActMaxTorchTransfBaseUni
from controversialstimuli.utility.plot import ImagePlotter, MoviePlotter
from controversialstimuli.analyses.confusion_matrix import (
    cluster_mtx,
    norm_conf_mtx,
)
from controversialstimuli.utility.plot import (
    get_main_diagonal_annotation,
    plot_conf_mtx_imgs,
)


LOGGER = logging.Logger(__name__)


class KMeansLogger(ABC):
    """Logger base class specifying the interface for logging kmeans results."""
    def __init__(self, list_of_gt_labels: List[int], output_folder: str = "kmeans_logger_output") -> None:
        """_summary_

        Args:
            list_of_gt_labels (List[int]): For each neuron, specify the cluster groundtruth index
        """
        self._list_of_gt_labels = list_of_gt_labels
        self._output_folder = output_folder

    @abstractmethod
    def log(self, cluster_assignments: Dict[int, List[int]], iteration: int) -> None:
        """Prints a log of the cluster assignments

        Args:
            cluster_assignments (Dict[int, List[int]]): Key: cluster index, Value: list of neuron indices
            iteration (int): Iteration number
        """
        pass

    @abstractmethod
    def plot_gt_confusion_matrix(self, cluster_assignments: Dict[int, List[int]], iteration: int) -> plt.Figure:
        """Plots the confusion matrix of the ground truth labels and the cluster assignments.

        Args:
            cluster_assignments (Dict[int, List[int]]): Key: cluster index, Value: list of neuron indices
            iteration (int): Iteration number

        Returns:
            plt.Figure: Figure object
        """
        pass


class DummyLogger(KMeansLogger):
    """Dummy logger that does not log or plot anything."""
    def __init__(self, *args, **kwargs):
        pass

    def log(self, *args, **kwargs):
        pass

    def plot_gt_confusion_matrix(self, *args, **kwargs):
        fig, _ = plt.subplots()
        return fig

    def spawn_for_folder(self, *args, **kwargs):
        logger = DummyLogger(*args, **kwargs)
        return logger


class DefaultLogger(KMeansLogger):
    def __init__(
        self,
        list_of_gt_labels: List[int],
        output_folder: str = "retina_logger_output",
        use_rgc_names: bool = False,
        plot: bool = False,
    ):
        """
        Args:
            use_rgc_names (bool, optional): Whether to convert the ground truth cluster indices to human readable 
                mouse RGC labels. Defaults to False.
            plot (bool, optional): Wheter to plot confusion matrices. Defaults to False.
        """
        super().__init__(list_of_gt_labels, output_folder)
        self._use_rgc_names = use_rgc_names
        self._plot = plot

    def spawn_for_folder(self, output_folder: str):
        logger = DefaultLogger(
            self._list_of_gt_labels, output_folder, use_rgc_names=self._use_rgc_names, plot=self._plot
        )
        return logger

    def log(self, cluster_assignments: Dict[int, List[int]], iteration: int):
        print(f"\nCluster Assignments for Iteration {iteration}:")
        for cluster_id, idc in cluster_assignments.items():
            counter = Counter()
            for idx in idc:
                gt_cluster_label = self._list_of_gt_labels[idx]
                counter[gt_cluster_label] += 1

            print(f"\nCluster {cluster_id}:")
            for group_id, count in counter.most_common(50):
                if self._use_rgc_names:
                    print(f"{RGC_GROUP_NAMES_DICT[group_id]}: {count}")
                else:
                    print(f"Labeled cluster {group_id}: {count}")
        print("\n")

        if self._plot and len(cluster_assignments) < 100:
            fig = self.plot_gt_confusion_matrix(cluster_assignments, iteration)
            path_to_save = f"{self._output_folder}/rgc_confusion_matrix_it_{iteration}.pdf"
            fig.savefig(path_to_save, dpi=300, bbox_inches="tight")
            plt.close("all")
        else:
            print(f"Too many clusters ({len(cluster_assignments)}), will not plot rgc matrix")

    def calculate_adjusted_rand_score(self, cluster_assignments: Dict[int, List[int]]) -> float:
        neuron_id_to_cluster_dict = {
            unit_idx: clust_id for clust_id, unit_idc in cluster_assignments.items() for unit_idx in unit_idc
        }
        neuron_id_to_gt_cluster_dict = {unit_idx: gt_id for unit_idx, gt_id in enumerate(self._list_of_gt_labels)}

        unit_ids = sorted(list(neuron_id_to_cluster_dict.keys()))
        neuron_id_to_cluster_list = [neuron_id_to_cluster_dict[i] for i in unit_ids]
        neuron_id_to_gt_cluster_list = [neuron_id_to_gt_cluster_dict[i] for i in unit_ids]

        score = adjusted_rand_score(neuron_id_to_cluster_list, neuron_id_to_gt_cluster_list)
        return score

    def plot_gt_confusion_matrix(
        self,
        cluster_assignments: Dict[int, List[int]],
        iteration: Optional[int] = None,
        normalize_gt_column_to_one: bool = True,
        normalize_cluster_row_to_one: bool = False,
        row_block_structure: bool = False,
        exclude_counts_below: int = 0,
        exclude_cluster_ids: Optional[List[int]] = None,
        annot: bool = False,
        norm_cbar: bool = False,
        overwrite_suptitle=None,
        cmap=None,
        overwrite_row_order: Optional[
            List[int]
        ] = None,
        cbar=True,
        figsize=(20, 10),
        paper=False,
        old2new_idc: Optional[list[int]]=None,
    ) -> plt.Figure:
        """Plots the confusion matrix of the ground truth labels and the cluster assignments.
        
        Args:
            cluster_assignments (Dict[int, List[int]]): Key: cluster index, Value: list of neuron indices
            iteration (Optional[int], optional): Iteration number. Defaults to None.
            normalize_gt_column_to_one (bool, optional): Normalize the ground truth columns to one. Defaults to True.
            normalize_cluster_row_to_one (bool, optional): Normalize the cluster rows to one. Defaults to False.
            row_block_structure (bool, optional): Sort the rows by block structure. Defaults to False.
            exclude_counts_below (int, optional): Exclude clusters with counts below this value. Defaults to 0.
            exclude_cluster_ids (Optional[List[int]], optional): List of cluster ids to exclude. Defaults to None.
            annot (bool, optional): Annotate the heatmap. Defaults to False.
            norm_cbar (bool, optional): Normalize the colorbar. Defaults to False.
            overwrite_suptitle ([type], optional): Overwrite the suptitle. Defaults to None.
            cmap ([type], optional): Colormap. Defaults to None.
            overwrite_row_order (Optional[List[int]], optional): right before plotting, after row_block_structure. Defaults to None.
            cbar (bool, optional): Show the colorbar. Defaults to True.
            figsize (Tuple[int, int], optional): Figure size. Defaults to (20, 10).
            paper (bool, optional): Use paper plotting style. Defaults to False.
            old2new_idc (Optional[list[int]], optional): Mapping from old (list index) to new (list values) cluster 
                indices. Defaults to None.
        """
        assert not (normalize_gt_column_to_one and normalize_cluster_row_to_one), "Only one normalization allowed"

        if max(cluster_assignments.keys()) + 1 != len(cluster_assignments):
            missing_keys = sorted(set(range(len(cluster_assignments))) - set(cluster_assignments.keys()))
            print("Missing keys in cluster_assignments, adding empty clusters with keys: ", missing_keys)
            cluster_assignments = {**cluster_assignments, **{k: [] for k in missing_keys}}

        exclude_cluster_ids = exclude_cluster_ids or []

        rand_score = self.calculate_adjusted_rand_score(cluster_assignments)
        cluster_counts_dict: Dict[int, Counter] = defaultdict(Counter)
        neurons_per_celltype = Counter()

        for cluster_id, idc in cluster_assignments.items():
            if len(idc) == 0:
                exclude_cluster_ids.append(cluster_id)
                for gt_cluster_label in self._list_of_gt_labels:
                    cluster_counts_dict[cluster_id][gt_cluster_label] = 0

            for idx in idc:
                gt_cluster_label = self._list_of_gt_labels[idx]
                neurons_per_celltype[gt_cluster_label] += 1
                cluster_counts_dict[cluster_id][gt_cluster_label] += 1

        neurons_per_cluster = {idx: sum(counter.values()) for idx, counter in cluster_counts_dict.items()}
        # Normalize clusters
        sorted_celltypes = sorted(neurons_per_celltype.keys())
        sorted_cluster_ids = sorted(cluster_counts_dict.keys())

        unnormalized_matrix = np.zeros((len(cluster_counts_dict), len(neurons_per_celltype)))
        for cluster_id, counter in cluster_counts_dict.items():
            for norm_id, gt_id in enumerate(sorted_celltypes):
                unnormalized_matrix[cluster_id, norm_id] = counter[gt_id]

        cluster_id_to_matrix_position = {c_id: m_id for m_id, c_id in enumerate(sorted_cluster_ids)}
        cluster_labels = np.array([f"Cluster {c_id} [{neurons_per_cluster[c_id]}]" for c_id in sorted_cluster_ids])

        exclude_mtx_idx = []
        if len(exclude_cluster_ids) > 0:
            exclude_mtx_idx.extend([cluster_id_to_matrix_position[c_id] for c_id in sorted(set(exclude_cluster_ids))])
        if exclude_counts_below > 0:
            exclude_mtx_idx.extend(
                [
                    cluster_id_to_matrix_position[c_id]
                    for c_id in sorted_cluster_ids
                    if neurons_per_cluster[c_id] < exclude_counts_below
                ]
            )
        if len(exclude_mtx_idx) > 0:
            unnormalized_matrix = np.delete(unnormalized_matrix, exclude_mtx_idx, axis=0)
            cluster_labels = np.delete(cluster_labels, exclude_mtx_idx, axis=0)

        normalized_matrix = norm_conf_mtx(
            unnormalized_matrix,
            norm_axis="row_l1" if normalize_cluster_row_to_one else "col_l1" if normalize_gt_column_to_one else None,
        )

        if row_block_structure:
            new_row_idc = cluster_mtx(normalized_matrix, metric="correlation", row_cluster=True, col_cluster=False)[1]
            normalized_matrix = normalized_matrix[new_row_idc]
            cluster_labels = cluster_labels[new_row_idc]

        if overwrite_row_order is not None:
            normalized_matrix = normalized_matrix[overwrite_row_order]
            cluster_labels = cluster_labels[overwrite_row_order]

        if old2new_idc:
            cluster_labels = [f"{old2new_idc[int(l.split()[1])]}\n[{l.split('[')[-1]}" for l in cluster_labels]

        if self._use_rgc_names:
            if not paper:
                rgc_labels = [
                    f"{g_id}: {RGC_GROUP_NAMES_DICT[g_id]} [{neurons_per_celltype[g_id]}]" for g_id in sorted_celltypes
                ]
            else:
                rgc_labels = [
                    f"{RGC_GROUP_NAMES_DICT[g_id]} [{neurons_per_celltype[g_id]}]" for g_id in sorted_celltypes
                ]
        else:
            rgc_labels = [f"Labeled cluster {g_id} [{neurons_per_celltype[g_id]}]" for g_id in sorted_celltypes]

        fig, axis = plt.subplots(figsize=figsize)
        seaborn.heatmap(
            data=normalized_matrix.T * 100,
            ax=axis,
            square=True,
            xticklabels=cluster_labels,
            yticklabels=rgc_labels,
            vmin=0.0 if norm_cbar else None,
            vmax=1.0 if norm_cbar else None,
            cbar=cbar,
            cbar_kws={"shrink": 0.1} if cbar else {},
            annot=annot,
            annot_kws={"fontsize": 5} if annot else {},
            fmt=".0f",
            cmap=cmap,
        )
        
        if old2new_idc:
            axis.set_xticklabels(axis.get_xticklabels(), rotation=0, fontsize=7)

        if not paper:
            axis.set_ylabel("Clusters")
            axis.set_xlabel("RGC Groups" if self._use_rgc_names else "Ground Truth Cluster Labels")
        else:
            axis.set_ylabel('Types from Baden et al. (2016)\n[# neurons]')
            axis.set_xlabel('Clusters\n[# neurons]')

        if iteration is None:
            iteration_string = ""
        else:
            iteration_string = f" in iteration {iteration}"

        if normalize_cluster_row_to_one:
            normalization_string = " (normalized cluster rows to one)"
        elif normalize_gt_column_to_one:
            normalization_string = " (normalized ground truth columns to one)"
        else:
            normalization_string = ""

        if not paper:
            fig.suptitle(
                overwrite_suptitle
                or f"Counts of groups for each cluster{iteration_string}{normalization_string} (ARI: {rand_score:.3f})"
            )

        return fig


class KMeans:
    def __init__(
        self,
        img_optimizer: ActMaxTorchTransfBaseUni,
        unit_idc: List[int],
        cluster_idc: List[int],  # ALL CLUSTER INDICES, e.g. list(range(num_clusters))
        optim_cluster_subset_idc: Optional[List[int]] = None,  # CLUSTER INDICES TO OPTIMIZE
        init_cluster_assignments: Optional[Dict[int, List[int]]] = None,  # maps cluster index to neuron index
        seed: Optional[int] = 42,
        max_iter: Optional[int] = None,
        reinitialize_stimuli: bool = False,
        verbose: bool = False,
        device: str = "cuda",
        logger: KMeansLogger = DummyLogger(),
        disable_progress_bar: bool = False,
        output_folder: str = "./kmeans_results",
        list_of_gt_labels: Optional[List[int]] = None,
        plot: bool = False,
    ):
        """Initializes a KMeans object for clustering neurons.

        Args:
            img_optimizer (ActMaxTorchTransfBaseUni): The image optimizer used for optimization.
            unit_idc (List[int]): The indices of the units (neurons) to be clustered. Must match the model output.
            cluster_idc (List[int]): All cluster indices. For example, `list(range(num_clusters))`.
            optim_cluster_subset_idc (Optional[List[int]]): The cluster indices to optimize. Defaults to None, which 
                means all clusters will be optimized.
            init_cluster_assignments (Optional[Dict[int, List[int]]]): Initial cluster assignments. Maps cluster index
                 (key) to neuron index (values). Defaults to None.
            seed (Optional[int]): The random seed for the initial cluster assignment. Defaults to 42.
            max_iter (Optional[int]): The maximum number of kmeans iterations. Defaults to None.
            reinitialize_stimuli (bool): Whether to reinitialize stimuli during clustering. Defaults to False.
            verbose (bool): Whether to print verbose output. Defaults to False.
            device (str): The device to use for computation. Defaults to "cuda".
            logger (KMeansLogger): The logger for logging clustering progress. Defaults to DummyLogger().
            disable_progress_bar (bool): Whether to disable the progress bar. Defaults to False.
            output_folder (str): The folder path to save clustering results. Defaults to "./kmeans_results".
            list_of_gt_labels (Optional[List[int]]): The list of ground truth labels for on the fly logging / confusion
                matrix plotting. Defaults to None.
            plot (bool): Whether to plot the clustering results. Defaults to False.
        """
        
        self._device = device
        self._img_optimizer = img_optimizer
        self._max_iter = max_iter
        self._unit_idc = unit_idc
        self._plot = plot
        self._reinitialize_stimuli = reinitialize_stimuli
        self._logger = logger
        self._list_of_gt_labels = list_of_gt_labels
        self._disable_progress_bar = disable_progress_bar
        self._iteration = 0

        self._verbose = verbose
        self._verbose_print = get_verbose_print_fct(verbose)

        self._output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)

        self._seed = seed
        if self._seed is not None:
            random.seed(self._seed)

        # Attributes that'll change during clustering
        self._cluster_img_dict: Dict[int, torch.FloatTensor] = {}  # cluster_id -> group_img
        self._cluster_assignments = self._init_cluster_assignments(
            init_cluster_assignments,
            cluster_idc,
        )
        
        self._logger.log(self._cluster_assignments, iteration=0)
        self._optim_cluster_subset_idc = sorted(optim_cluster_subset_idc or cluster_idc)

        self._verbose_print(
            f"Starting clustering for {self._num_units} neurons into {self._num_clusters} clusters (or less if optim_cluster_subset_idc passed)"
        )

    def _init_cluster_assignments(self, init_cluster_assignments: Optional[Dict[int, List[int]]], cluster_idc: List[int]) -> Dict[int, List[int]]:
        """
        Initializes the cluster assignments for the K-means algorithm.

        Args:
            init_cluster_assignments (Optional[Dict[int, List[int]]]): A dictionary containing the initial cluster 
                assignments, mapping cluster index (key) to neuron index (values). If None, the cluster assignments
                will be initialized randomly.
            cluster_idc (List[int]): A list of all cluster indices. For example, `list(range(num_clusters))`.

        Returns:
            Dict[int, List[int]]: A dictionary containing the cluster assignments, where the keys are cluster 
            indices and the values are lists of unit indices assigned to each cluster.
        """
        shuffled_unit_idc = copy.deepcopy(self._unit_idc)
        random.Random(self._seed).shuffle(shuffled_unit_idc)

        if init_cluster_assignments is None:
            logging.warning(
                "No init cluster assignments provided, initializing randomly ignoring optim_cluster_subset_idc"
            )
            elements_per_cluster = ceil(self._num_units / len(cluster_idc))
            init_cluster_assignments = dict()
            for clust_id in cluster_idc:
                start_idx = clust_id * elements_per_cluster
                end_idx = (clust_id + 1) * elements_per_cluster
                init_cluster_assignments[clust_id] = shuffled_unit_idc[start_idx:end_idx]
                # because of ceil, the end index for the last cluster might be larger than the length of the list, but 
                # that will not cause an error

        return copy.deepcopy(init_cluster_assignments)

    def get_neuron_to_cluster_id(self) -> Dict[int, int]:
        """Returns a dictionary mapping neuron indices (keys) to cluster indices (value)."""
        neuron_id_to_cluster_dict = {
            unit_idx: clust_id for clust_id, unit_idc in self._cluster_assignments.items() for unit_idx in unit_idc
        }
        return neuron_id_to_cluster_dict

    @property
    def _num_clusters(self) -> int:
        return len(self._cluster_idc)

    @property
    def _num_clusters_to_optimize(self) -> int:
        return len(self._optim_cluster_subset_idc)

    @property
    def _num_units(self) -> int:
        return len(self._unit_idc)

    @property
    def _cluster_idc(self) -> List[int]:
        """Returns a sorted list of all cluster indices, lenth is number of clusters"""
        return sorted(self._cluster_assignments.keys())

    @property
    def _unit_idc_to_optimize(self) -> List[int]:
        return [
            unit_idx for clust_id in self._optim_cluster_subset_idc for unit_idx in self._cluster_assignments[clust_id]
        ]

    def remove_empty_clusters(self) -> int:
        """Remove empty clusters and returns the number of removed clusters"""
        cluster_is_not_empty = np.array([len(self._cluster_assignments[k]) > 0 for k in self._cluster_idc])
        non_empty_cumsum = cluster_is_not_empty.cumsum()

        num_clusters_to_remove = self._num_clusters - np.sum(cluster_is_not_empty)

        if num_clusters_to_remove > 0:
            assert (
                self._num_clusters == self._num_clusters_to_optimize
            ), "Cluster removal not tested when only clustering on subset of clusters"
            old_to_new_cluster_id_map = {
                old_id: cumsum - 1 for old_id, cumsum in enumerate(non_empty_cumsum) if cluster_is_not_empty[old_id] > 0
            }

            new_assignments = {
                old_to_new_cluster_id_map[c_id]: n_id_list
                for c_id, n_id_list in self._cluster_assignments.items()
                if c_id
                in old_to_new_cluster_id_map  # only non empty clusters are present in the old_to_new_cluster_id_map
            }
            new_img_dict = {
                old_to_new_cluster_id_map[c_id]: img
                for c_id, img in self._cluster_img_dict.items()
                if c_id in old_to_new_cluster_id_map
            }

            self._cluster_assignments = new_assignments
            self._cluster_img_dict = new_img_dict
            self._optim_cluster_subset_idc = self._cluster_idc

        return num_clusters_to_remove

    def run(self) -> bool:
        """Returns: True if at any iteration a neuron changed its cluster assignment."""
        self._verbose_print(f"Running clustering with {self._num_clusters} clusters")

        changed_at_any_iter = False
        while self._max_iter is None or self._iteration < self._max_iter:
            # Max iter end condition
            if self._max_iter is not None and self._iteration > self._max_iter:
                self._verbose_print(f"Reached max iteration {self._max_iter}, exiting")
                return changed_at_any_iter

            print(f"\n\n# knn iteration: {self._iteration}\n")

            num_removed_empty_clusters = self.remove_empty_clusters()
            if num_removed_empty_clusters > 0:
                print(f"Removed {num_removed_empty_clusters} empty clusters, {self._num_clusters} clusters remaining.")

            if self._num_clusters <= 1:
                print(f"Only {self._num_clusters} remaining, stopping clustering")
                return changed_at_any_iter

            self._optim_images(
                reinitialize_stimuli=self._reinitialize_stimuli
            )  # changes (among others): self._clust_img_dict

            if self._plot and self._num_clusters < 100:
                self.plot_response_confusion_matrix(
                    f"{self._output_folder}/response_confusion_matrix_{self._iteration:02d}.pdf",
                    title=f"Iteration {self._iteration}",
                )
                plt.close("all")
            else:
                print(f"Too large number of clusters ({self._num_clusters}), not plotting response matrix")

            changed_neuron_assignments = self._reassign_units(remove_empty_clusters=True)
            changed_at_any_iter = changed_at_any_iter or bool(changed_neuron_assignments)

            self.save_cluster_results_to_folder()
            self._logger.log(self._cluster_assignments, iteration=self._iteration)

            # No neuron changed cluster end condition
            if changed_neuron_assignments == 0:
                self._verbose_print("No neurons changed clusters, exiting")
                return changed_at_any_iter
            
            self._iteration += 1

        # Max iter end condition
        self._verbose_print(f"Reached max iteration {self._max_iter}, exiting")
        return changed_at_any_iter

    def evaluate_training_loss(self, return_list: bool = False) -> Union[float, list[float]]:
        """Evaluates the MDS training loss for each MDS/cluster in the K-means algorithm.

        Args:
            return_list (bool, optional): If True, returns a list of losses for each cluster. If False, returns the
                average loss across all clusters. Defaults to False.

        Returns:
            Union[float, list[float]]: The average loss across all clusters if return_list is False. A list of 
                losses for each cluster if return_list is True.
        """
        list_of_losses = []
        for c_id in self._cluster_idc:
            stimulus = self._cluster_img_dict[c_id]
            self._img_optimizer.reset(stimulus)
            self._img_optimizer.objective_fct.set_clusters(
                on_clust_idx=c_id,
                off_clust_idc=list(set(self._cluster_idc) - set([c_id])),
                clust_assignments=self._neuron_id2cluster_np,
                unit_idc=self._unit_idc,
            )
            self._img_optimizer.objective_fct = self._img_optimizer.objective_fct.to(self._device)
            loss = self._img_optimizer.get_loss().item()
            list_of_losses.append(loss)

        if return_list:
            return list_of_losses
        
        avg_loss = np.mean(list_of_losses)
        return avg_loss

    def run_split_cluster(
            self,
            num_clusters: int,
            use_rgc_init: bool,
            optim_steps_to_estimate_loss: int,
            split_max_iter: Optional[int] = None,
            subcluster_kmeans_max_iter: int = 50,
            subcluster_kmeans_img_optimizer_max_iter: int = 1,
            use_new_seed: bool = False,
            use_self_plot: bool = False,
    ) -> bool:
        """Splits the clusters into new clusters.

        Args:
            num_clusters (int): Number of new clusters to split in.
            use_rgc_init (bool): Whether to use RGC initialization.
            optim_steps_to_estimate_loss (int): Number of optimization steps to estimate loss.
            split_max_iter (Optional[int], optional): Maximum number of iterations for splitting. Defaults to None.
            subcluster_kmeans_max_iter (int, optional): Maximum number of iterations for subcluster k-means. Defaults to 50.
            subcluster_kmeans_img_optimizer_max_iter (int, optional): Maximum number of iterations for subcluster k-means image optimizer. Defaults to 1.
            use_new_seed (bool, optional): Whether to use a new seed. Defaults to False.
            use_self_plot (bool, optional): Whether to follow the self.plot argument for the subset clustering. 
                Defaults to False, disabling subset cluster plotting.

        Returns:
            bool: True if clusters were changed, False otherwise.
        """
        print(f"splitting into {num_clusters} new clusters")
        clusters_changed = True
        iteration = 0

        while clusters_changed:
            print(f"SPLIT CLUSTER iteration {iteration}")

            if split_max_iter is not None and iteration == split_max_iter:
                print(f"Reached max iteration {split_max_iter}, exiting")
                return clusters_changed
            
            iteration += 1
            clusters_changed = False
            cluster_id_list = self._cluster_idc

            for id_to_split in cluster_id_list:
                if id_to_split not in self._cluster_idc:
                    print(
                        f"Some clusters were removed as {id_to_split} is not in cluster ids {self._cluster_idc}"
                    )
                    continue

                print(f"Trying to split cluster {id_to_split}")
                success, new_loss, new_idc = self.split_cluster_if_loss_improves(
                    id_to_split,
                    num_clusters,
                    use_rgc_init=use_rgc_init,
                    optim_steps_to_estimate_loss=optim_steps_to_estimate_loss,
                    subcluster_kmeans_max_iter=subcluster_kmeans_max_iter,
                    subcluster_kmeans_img_optimizer_max_iter=subcluster_kmeans_img_optimizer_max_iter,
                    use_new_seed=use_new_seed,
                    use_self_plot=use_self_plot,

                )
                print(
                    f"Split of cluster {id_to_split} resulted in these new clusters: {new_idc}, new_loss: {new_loss:.5f}"
                )
                clusters_changed |= success

            # save full state
            date_str = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
            subdir_path = f"{self._output_folder}/split_full_states"
            os.makedirs(subdir_path, exist_ok=True)
            self.save_cluster_results_to_folder(
                folder_path=subdir_path, file_name=f"full_state_{date_str}_after_split_iteration_{iteration}.pkl"
            )

            if self._plot:
                self.plot_response_confusion_matrix(
                    f"{subdir_path}/response_confusion_matrix_after_split_iteration_{iteration}.pdf",
                    title="After split",
                )
                plt.close("all")
                
                fig = self._logger.plot_gt_confusion_matrix(self._cluster_assignments, 0)
                fig.savefig(
                    f"{subdir_path}/rgc_confusion_matrix_after_split_iteration_{iteration}.pdf",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close("all")

        return clusters_changed

    def split_cluster_if_loss_improves(
        self,
        cluster_id: int,
        num_subclusters: int,
        use_rgc_init: bool,
        optim_steps_to_estimate_loss: int = 20,
        subcluster_kmeans_max_iter: int = 50,
        subcluster_kmeans_img_optimizer_max_iter: int = 1,
        use_new_seed: bool = False,
        use_self_plot: bool = False,
    ) -> Tuple[bool, float, Set[int]]:
        """Splits a cluster into subclusters if the loss improves.

        Args:
            cluster_id (int): The ID of the cluster to be split.
            num_subclusters (int): The number of subclusters to create.
            use_rgc_init (bool): Whether to use mouse RGC initialization for subclusters.
            optim_steps_to_estimate_loss (int, optional): The number of stimulus optimization steps to estimate the 
                loss. Defaults to 20.
            subcluster_kmeans_max_iter (int, optional): The maximum number of iterations for subcluster k-means.
                Defaults to 50.
            subcluster_kmeans_img_optimizer_max_iter (int, optional): The maximum number of iterations to optimize the
                stimulus during one k-means iteration. Defaults to 1.
            use_new_seed (bool, optional): Whether to use a new seed for subclusters. Defaults to False.
            use_self_plot (bool, optional): Whether to follow the self._plot attribute for plotting. Defaults to False,
                then self._plot is being ignored and plotting disabled.

        Returns:
            Tuple[bool, float, Set[int]]: A tuple containing the success status of the split (True if successful, 
                False otherwise), the new loss value, and the IDs of the new subclusters.

        Raises:
            AssertionError: If the number of clusters does not match the number of cluster image dictionary keys.
        """
        date_str = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        subdir = f"split_{date_str}_cluster_{cluster_id}"
        subdir_path = f"{self._output_folder}/{subdir}"
        os.makedirs(subdir_path, exist_ok=True)

        optim_max_iter = self._img_optimizer.max_iter
        kmeans_state_dict_before_split = self.cluster_results_to_dict()
        assert self._num_clusters == len(self._cluster_img_dict.keys())

        # get previous loss
        self._img_optimizer.max_iter = optim_steps_to_estimate_loss
        self._optim_images(reinitialize_stimuli=True)
        previous_loss = self.evaluate_training_loss()

        if self._plot:
            self.plot_response_confusion_matrix(
                f"{subdir_path}/response_confusion_matrix_before_split.pdf",
                title=f"Before split, loss: {previous_loss:.5f}",
            )
            plt.close("all")
            fig = self._logger.plot_gt_confusion_matrix(self._cluster_assignments, 0)
            fig.savefig(
                f"{subdir_path}/rgc_confusion_matrix_before_split.pdf",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close("all")

        # Run subclustering
        # hacky, the local kmeans uses the same optimizer
        ids_of_cluster_to_split = self._cluster_assignments[cluster_id]
        self._img_optimizer.max_iter = subcluster_kmeans_img_optimizer_max_iter
        cluster_idc_of_local_kmeans = list(range(num_subclusters))

        local_kmeans = self.spawn_for_subset(
            unit_idc=ids_of_cluster_to_split,
            cluster_idc=cluster_idc_of_local_kmeans,
            subdirectory_name=subdir,
            max_iter=subcluster_kmeans_max_iter,
            use_rgc_init=use_rgc_init,
            use_new_seed=use_new_seed,
            use_self_plot=use_self_plot,
        )
        local_kmeans.run()
        if local_kmeans._num_clusters <= 1:
            print(f"No new subclusters found during splitting cluster {cluster_id}")
            return False, float("inf"), set()

        # Add new clusters to this kmeans object
        # Keep idc continuous and overwrite the cluster_id with local cluster 0,
        # and create new clusters for local clusters ids 1, ...
        local_to_global_id_map = {i: (self._num_clusters - 1) + i for i in range(1, num_subclusters)}
        local_to_global_id_map[0] = cluster_id

        for local_c_idx in local_kmeans._cluster_idc:
            stimulus = local_kmeans._cluster_img_dict[local_c_idx]
            global_c_idx = local_to_global_id_map[local_c_idx]
            self._cluster_img_dict[global_c_idx] = stimulus
            self._cluster_assignments[global_c_idx] = local_kmeans._cluster_assignments[local_c_idx]
        self._optim_cluster_subset_idc = list(self._cluster_img_dict.keys())

        # Optimize images and reassign
        # if you immediately reassign you only get a few neurons for the new clusters, as their stimuli are not 
        # optimized heavily yet
        self._img_optimizer.max_iter = optim_steps_to_estimate_loss
        self._optim_images(reinitialize_stimuli=True)
        self._reassign_units(remove_empty_clusters=True)
        self._optim_images(reinitialize_stimuli=True)

        # evaluate loss
        new_loss = self.evaluate_training_loss()
        successful_split = new_loss < previous_loss
        postfix_str = "successful" if successful_split else "unsuccessful"

        assert self._num_clusters == len(self._cluster_img_dict.keys())

        if self._plot:
            # Plot state after clustering
            self.plot_response_confusion_matrix(
                f"{subdir_path}/response_confusion_matrix_after_{postfix_str}_split.pdf",
                title=f"{postfix_str} split, loss={new_loss:.5f}",
            )
            plt.close("all")
            fig = self._logger.plot_gt_confusion_matrix(self._cluster_assignments, 1)

            fig.savefig(
                f"{subdir_path}/rgc_confusion_matrix_after_{postfix_str}_split.pdf",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close("all")

        print(f"Successful split: {successful_split}, prev_loss {previous_loss:.5f}, new loss {new_loss:.5f}")
        self._img_optimizer.max_iter = optim_max_iter

        if successful_split:
            new_cluster_idc = set(local_to_global_id_map.values())
        else:
            new_cluster_idc = set()
            # revert cluster assignment
            self.load_cluster_results_from_dict(kmeans_state_dict_before_split)

        return successful_split, new_loss, new_cluster_idc

    def _optim_images(self, reinitialize_stimuli: bool) -> None:
        """Optimizes the images for each cluster.

        Args:
            reinitialize_stimuli (bool): Whether to reinitialize the stimuli before optimization or start optimization
                from the previous stimulus.
        """
        list_of_losses = []
        for clust_id in tqdm(self._optim_cluster_subset_idc, disable=self._disable_progress_bar):
            if reinitialize_stimuli:
                initial_stimulus: Optional[torch.FloatTensor] = None
            else:
                initial_stimulus = self._cluster_img_dict.get(clust_id, None)
            self._img_optimizer.reset(initial_stimulus)

            off_clust_idc = list(set(self._cluster_idc) - set([clust_id]))
            self._img_optimizer.objective_fct.set_clusters(
                on_clust_idx=clust_id,
                off_clust_idc=off_clust_idc,
                clust_assignments=self._neuron_id2cluster_np,
                unit_idc=self._unit_idc,
            )
            self._img_optimizer.objective_fct = self._img_optimizer.objective_fct.to(self._device)

            clust_members = self._cluster_assignments[clust_id]
            if len(clust_members) == 0:
                self._verbose_print(f"Cluster {clust_id} has no members, skipping")
                continue

            start_time = time.time()

            loss = self._img_optimizer.maximize()
            list_of_losses.append(loss)
            self._cluster_img_dict[clust_id] = self._img_optimizer.img.detach()

            self._verbose_print(
                f"Generated group image for group {clust_id} with {len(clust_members)} members "
                f"in {time.time() - start_time:.1f}s"
            )
        print(f"\nMean Loss over {len(list_of_losses)} clusters: {np.mean(list_of_losses):.5f}")
        print(f"Per cluster losses: {list_of_losses}")

    def _predict_cluster_idc(self) -> Dict[int, int]:
        """Predicts the cluster index for each neuron.

        Returns:
            Dict[int, int]: A dictionary mapping neuron indices to cluster indices.
        """
        resp = self._predict().detach().cpu().numpy()

        if len(self._unit_idc) != resp.shape[1]:
            raise NotImplementedError("Unit_idc must match the model output length")
        if len(set(self._unit_idc)) != len(self._unit_idc):
            raise NotImplementedError("Unit_idc must be consecutive")

        clust_idc_pred = np.array(self._optim_cluster_subset_idc)[np.argmax(resp, 0)]
        
        assert min(self._unit_idc) >= 0
        assert len(clust_idc_pred) == len(
            self._unit_idc
        ), f"len(clust_idc_pred) = {len(clust_idc_pred)} != {len(self._unit_idc)}"

        neuron_id2cluster_dict = self.get_neuron_to_cluster_id()
        neuron_to_cluster = {
            unit_id: clust_idc_pred[idx]
            for idx, unit_id in enumerate(self._unit_idc)
            if neuron_id2cluster_dict[unit_id] in self._optim_cluster_subset_idc
        }
        return neuron_to_cluster

    def _predict(self) -> torch.Tensor:
        """Returns: (len(self._clust_idc), len(self._unit_idc)) shaped tensor of model responses to the cluster stimuli."""
        responses = []
        unit_idc = torch.tensor(self._unit_idc, requires_grad=False)

        for i in self._optim_cluster_subset_idc:
            stimulus = self._cluster_img_dict[i]
            stimulus_cuda = stimulus.to(self._device)
            full_resp = self._img_optimizer.model(stimulus_cuda).detach().cpu()
            resp = full_resp[:, unit_idc]
            responses.append(resp)
        responses = torch.cat(responses)
        return responses

    def predict_traces(self) -> torch.Tensor:
        """Predict full response traces."""
        responses = []
        unit_idc = torch.tensor(self._unit_idc, requires_grad=False)
        for i in self._optim_cluster_subset_idc:
            stimulus = self._cluster_img_dict[i]
            stimulus_cuda = stimulus.to(self._device)
            full_traces = self._img_optimizer.model.forward_traces(stimulus_cuda).detach().cpu()
            resp = full_traces[:, :, unit_idc]
            responses.append(resp)
        responses = torch.cat(responses)
        return responses

    def _reassign_units(self, remove_empty_clusters: bool = True) -> int:
        """Reassigns the units to the clusters based on the optimized stimuli.

        Returns:
            int: The number of neurons that changed their cluster assignment.
        """
        start_time = time.time()

        cluster_idc_pred = self._predict_cluster_idc()
        changed_neuron_assignments = 0
        new_cluster_assignments: Dict[int, List[int]] = defaultdict(list)
        old_neuron_id_to_cluster_dict = self.get_neuron_to_cluster_id()

        for unit_idx, new_cluster_id in cluster_idc_pred.items():
            new_cluster_assignments[new_cluster_id].append(unit_idx)
            old_cluster_id = old_neuron_id_to_cluster_dict[unit_idx]
            if new_cluster_id != old_cluster_id:
                changed_neuron_assignments += 1

        # this code will also update clusters no neurons got assigned to
        for cluster_id in self._optim_cluster_subset_idc:
            self._cluster_assignments[cluster_id] = new_cluster_assignments[cluster_id]
        if remove_empty_clusters:
            self.remove_empty_clusters()
        self._verbose_print(f"Assigned {self._num_units} neurons to clusters in {time.time() - start_time:.1f}s")

        cluster_lengths = {clust_id: len(unit_idc) for clust_id, unit_idc in self._cluster_assignments.items()}
        self._verbose_print(f"Cluster sizes: {cluster_lengths}")
        self._verbose_print(f"{changed_neuron_assignments} neurons changed clusters")
        return changed_neuron_assignments

    @property
    def _neuron_id2cluster_np(self):
        """Returns an array of cluster assignments for each neuron ID."""
        neuron_id2cluster_dict = self.get_neuron_to_cluster_id()
        full_clust_assignments = np.array([neuron_id2cluster_dict[n] for n in sorted(self._unit_idc)])
        return full_clust_assignments

    def cluster_results_to_dict(self) -> Dict[str, Any]:
        stimuli_list = [
            self._cluster_img_dict[idx].detach().cpu().numpy() for idx in sorted(self._cluster_img_dict.keys())
        ]
        stimuli_np = np.concatenate(stimuli_list)
        data_dict = {
            "cluster_assignments": copy.deepcopy(self._cluster_assignments),
            "unit_idc": copy.deepcopy(self._unit_idc),
            "number_of_clusters": self._num_clusters,
            "seed": self._seed,
            "max_iter": self._max_iter,
            "iteration": self._iteration,
            "stimuli_numpy": stimuli_np,
        }
        return data_dict

    def save_cluster_results_to_folder(self, folder_path: Optional[str] = None, file_name: Optional[str] = None):
        folder_path = folder_path or self._output_folder
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        data_dict = self.cluster_results_to_dict()
        pickle_dump(
            folder_path,
            f"kmeans_cluster_iteration_{self._iteration:03d}_results.pkl" if file_name is None else file_name,
            data_dict,
        )

    def load_cluster_results_from_dict(self, data_dict: Dict[str, Any]) -> None:
        self._cluster_assignments = data_dict["cluster_assignments"]
        self._optim_cluster_subset_idc = self._cluster_idc
        self._iteration = data_dict.get("iteration", 0)
        stimuli_np = data_dict.get("stimuli_numpy", None)

        if stimuli_np is not None:
            assert self._num_clusters == stimuli_np.shape[0]
            print(f"Loading {stimuli_np.shape[0]} stimuli")
            self._cluster_img_dict = {}
            for idx in range(stimuli_np.shape[0]):
                stimulus = np.expand_dims(stimuli_np[idx], 0)
                stimulus_torch = torch.tensor(stimulus).to(self._device)
                self._cluster_img_dict[idx] = stimulus_torch

    def load_cluster_results_from_file(self, path: str) -> None:
        with open(path, "rb") as f:
            data_dict = pickle.load(f)
        self.load_cluster_results_from_dict(data_dict)

    def load_stimuli(self, prefix: str) -> None:
        for idx in range(self._num_clusters):
            path = f"{prefix}_{idx:03d}.npy"
            img = np.load(path)
            img_torch = torch.tensor(img).to(self._device)
            self._cluster_img_dict[idx] = img_torch

    def spawn_for_subset(
        self,
        unit_idc: List[int],
        cluster_idc: List[int],
        subdirectory_name: str,
        max_iter: int = 10,
        use_rgc_init: bool = False,
        use_new_seed: bool = False,
        use_self_plot: bool = False,
    ):  # -> Kmeans
        """Spawns a new KMeans object for a subset of clusters.

        Args:
            unit_idc (List[int]): The indices of the units to optimize.
            cluster_idc (List[int]): The indices of the clusters to optimize.
            subdirectory_name (str): The name of the subdirectory to save the results.
            max_iter (int, optional): The maximum number of iterations for the KMeans algorithm. Defaults to 10.
            use_rgc_init (bool, optional): Whether to use mouse RGC initialization. Defaults to False.
            use_new_seed (bool, optional): Whether to use a new seed. Defaults to False.
            use_self_plot (bool, optional): Whether to follow the self.plot attribute for plotting. Defaults to False,
                then self._plot is being ignored and plotting disabled.
        
        Returns:
            Kmeans: A new KMeans object for the subset of clusters.
        """
        new_output_folder = self._output_folder + "/" + subdirectory_name
        new_logger = self._logger.spawn_for_folder(new_output_folder)

        if use_rgc_init:
            assert self._list_of_gt_labels is not None, "Provide a neuron_list when requesting use_rgc_init"
            rgc_group_to_neuron_list = defaultdict(list)
            for idx in unit_idc:
                gt_label = self._list_of_gt_labels[idx]
                rgc_group_to_neuron_list[gt_label].append(idx)

            init_cluster_assignments = {
                idx: rgc_group_to_neuron_list[t] for idx, t in enumerate(rgc_group_to_neuron_list.keys())
            }
            cluster_idc = list(range(len(init_cluster_assignments)))
        else:
            init_cluster_assignments = None

        kmeans_object = KMeans(
            img_optimizer=self._img_optimizer,
            unit_idc=unit_idc,
            cluster_idc=cluster_idc,
            init_cluster_assignments=init_cluster_assignments,
            seed=self._seed + 1 if use_new_seed else None,
            max_iter=max_iter,
            reinitialize_stimuli=self._reinitialize_stimuli,
            verbose=self._verbose,
            device=self._device,
            logger=new_logger,
            disable_progress_bar=self._disable_progress_bar,
            output_folder=new_output_folder,
            list_of_gt_labels=self._list_of_gt_labels,
            plot=self._plot if use_self_plot else False,
        )
        return kmeans_object

    def plot_response_confusion_matrix(
        self,
        path_to_save: Optional[str] = None,
        title: Optional[str] = None,
        metric="correlation",
        cbar: bool = False,
        exclude_counts_below: int = 0,
        center: Optional[float] = None,
        norm_type: str = "column_diag",
        cmap=None,
        overwrite_row_order=None,
        paper=False,
        figsize=None,
        old2new_idc=None,
        vmin=None,
        vmax=None,
        annot=False,
        head_row=True,
        xlabel=None,
    ) -> plt.Figure:
        """Plots the response confusion matrix.

        Args:
            path_to_save (Optional[str], optional): The path to save the plot. Defaults to None.
            title (Optional[str], optional): The title of the plot. Defaults to None.
            metric (str, optional): The metric to use for clustering. Defaults to "correlation".
            cbar (bool, optional): Whether to show the colorbar. Defaults to False.
            exclude_counts_below (int, optional): Exclude clusters with less than this number of counts. Defaults to 0.
            center (Optional[float], optional): The center value for the colormap. Defaults to None.
            norm_type (str, optional): The normalization type. Defaults to "column_diag", 
                see ..analyses.confusion_matrix.norm_conf_mtx for all options.
            cmap ([type], optional): The colormap to use. Defaults to None.
            overwrite_row_order ([type], optional): Overwrite the row order right before plotting. Defaults to None.
            paper (bool, optional): Whether to use paper style plotting. Defaults to False.
            figsize ([type], optional): The figure size. Defaults to None.
            old2new_idc ([type], optional): A mapping from old to new indices for plotting. Defaults to None.
            vmin ([type], optional): The minimum value for the colormap. Defaults to None.
            vmax ([type], optional): The maximum value for the colormap. Defaults to None.
            annot (bool, optional): Whether to annotate the plot. Defaults to False.
            head_row (bool, optional): Whether to use the head row. Defaults to True.
            xlabel ([type], optional): The x-axis label. Defaults to None.

        Returns:
            plt.Figure: The plot figure.
        """
        if head_row is False:
            raise NotImplementedError("head_row=False not implemented yet")
        
        responses = self._predict()  # shape (len(self._clust_idc), len(self._unit_idc))
        confusion_mat = np.zeros((self._num_clusters, self._num_clusters))
        annotation_images_plotter_list = []
        clusters_to_delete = []

        for cluster_id in range(self._num_clusters):
            cluster_mask = torch.tensor(self._neuron_id2cluster_np == cluster_id)

            if torch.sum(cluster_mask) < exclude_counts_below:
                clusters_to_delete.append(cluster_id)

            if torch.any(cluster_mask):
                cluster_responses = responses[:, cluster_mask]
                mean_cluster_responses = torch.mean(cluster_responses, axis=-1)
                confusion_mat[:, cluster_id] = mean_cluster_responses.detach().cpu().numpy()
            else:
                confusion_mat[:, cluster_id] = np.nan
                clusters_to_delete.append(cluster_id)

            stimulus = self._cluster_img_dict[cluster_id].squeeze(axis=0).detach().cpu().numpy()
            stimuli_plotter_class = MoviePlotter if len(self._img_optimizer.canvas_size) >= 4 else ImagePlotter
            stimulus_plotter = stimuli_plotter_class(stimulus)
            annotation_images_plotter_list.append(stimulus_plotter)

        confusion_mat = np.delete(confusion_mat, clusters_to_delete, axis=0)
        confusion_mat = np.delete(confusion_mat, clusters_to_delete, axis=1)
        annotation_images_plotter_list = [
            a for i, a in enumerate(annotation_images_plotter_list) if i not in clusters_to_delete
        ]
        group_labels = [
            f"{old2new_idc[i] if old2new_idc else i}\n[{len(self._cluster_assignments[i])}]"
            for i in range(self._num_clusters)
            if i not in clusters_to_delete
        ]
        mds_labels = [
            f"{old2new_idc[i] if old2new_idc else i}"
            for i in range(self._num_clusters)
            if i not in clusters_to_delete
        ]

        conf_mtx = norm_conf_mtx(confusion_mat, [norm_type])
        annot = get_main_diagonal_annotation(conf_mtx)
        conf_mtx, col_idc = cluster_mtx(conf_mtx, metric=metric)

        annot = annot[col_idc]
        annot = annot[:, col_idc]
        annotation_plotter_list_sorted = [annotation_images_plotter_list[i] for i in col_idc]
        group_labels_sorted = [group_labels[i] for i in col_idc]
        mds_labels_sorted = [mds_labels[i] for i in col_idc]

        if overwrite_row_order is not None:
            conf_mtx = conf_mtx[overwrite_row_order]
            conf_mtx = conf_mtx[:, overwrite_row_order]
            annot = annot[overwrite_row_order]
            annot = annot[:, overwrite_row_order]
            annotation_plotter_list_sorted = [annotation_plotter_list_sorted[i] for i in overwrite_row_order]
            group_labels_sorted = [group_labels_sorted[i] for i in overwrite_row_order]
            mds_labels_sorted = [mds_labels_sorted[i] for i in overwrite_row_order]

        if title is None:
            title = path_to_save

        fig = plot_conf_mtx_imgs(
            conf_mtx * 100 if paper else conf_mtx,
            annot=annot if not paper else True, 
            fmt=".0f" if paper else "",
            h_plotter_list=annotation_plotter_list_sorted,
            v_plotter_list=annotation_plotter_list_sorted,
            title=f"Response confusion matrix: {title}" if not paper else "",
            yticklabels=mds_labels_sorted,
            xticklabels=group_labels_sorted,
            cbar=cbar,
            center=center,
            figsize_scale=2.0,
            cmap=cmap,
            figsize=figsize,
            paper=paper,
            vmin=vmin,
            vmax=vmax,
            xlabel=xlabel,
        )

        if path_to_save:
            fig.savefig(path_to_save, dpi=300, bbox_inches="tight")
            plt.close("all")
        
        return fig

    def get_plotter_list(self, normalize_temporal_components_independently: bool = False) -> List[Any]:
        """Returns a list of plotter objects for each cluster in the k-means clustering, 
            used in plot_traces_grouped_by_rgc_cells.ipynb.

        Args:
            normalize_temporal_components_independently (bool, optional): Flag indicating whether to normalize the temporal components independently. Defaults to False.

        Returns:
            List[Any]: A list of plotter objects for each cluster.
        """
        annotation_images_plotter_list = []
        for cluster_id in range(self._num_clusters): 
            stimulus = self._cluster_img_dict[cluster_id].squeeze(axis=0).detach().cpu().numpy()
            stimuli_plotter_class = MoviePlotter if len(self._img_optimizer.canvas_size) >= 4 else ImagePlotter
            stimulus_plotter = stimuli_plotter_class(stimulus)
            annotation_images_plotter_list.append(stimulus_plotter)

        if normalize_temporal_components_independently:
            max_uv = max(p.abs_max_uv() for p in annotation_images_plotter_list)
            max_green = max(p.abs_max_green() for p in annotation_images_plotter_list)
            for p in annotation_images_plotter_list:
                p.norm_uv(max_uv)
                p.norm_green(max_green)

        return annotation_images_plotter_list

