"""Computes confusion matrices for cluster stimuli and analyze those. Plotting functions can be found in ..utility.plot"""

import warnings
from typing import Dict, Iterable, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from neuralpredictors.training import eval_state
from sklearn.cluster import SpectralBiclustering, SpectralCoclustering

from .model import norm_unit_act_imgs, pred_max_rot


def cluster_conf_mtx(
    conf_mtx: np.ndarray,
    n_clusters: int,
    optimizer_cls: Union[SpectralBiclustering, SpectralCoclustering],
    random_state: int = 1000,
):
    """Cluster the confusion matrix.

    Args:
        conf_mtx: confusion matrix
        n_clusters: number of clusters
        optimizer_cls: class of optimizer to use, either SpectralBiclustering or SpectralCoclustering

    Returns:
        fit_data: clustered confusion matrix
        row_labels: cluster labels for rows (i.e. for each row an index of the cluster to which the row corresponds)
        col_labels: cluster labels for columns (i.e. for each column an index of the cluster to which the column corresponds)
    """

    if optimizer_cls not in [SpectralBiclustering, SpectralCoclustering]:
        raise ValueError(
            f"Invalid optimizer class: {optimizer_cls}. Must be either SpectralBiclustering or SpectralCoclustering."
        )

    model = optimizer_cls(
        n_clusters=n_clusters, svd_method="arpack", random_state=random_state
    )
    model.fit(conf_mtx)

    fit_data = conf_mtx[np.argsort(model.row_labels_)]
    fit_data = fit_data[:, np.argsort(model.column_labels_)]

    return fit_data, model.row_labels_, model.column_labels_


def cluster_mtx(
    mtx: np.ndarray,
    metric: str = "euclidean",
    col_cluster: bool = True,
    row_cluster: bool = False,
):
    """[summary]

    Args:
        mtx (2d array)
        metric: correlation and euclidean yielded reasonable results
        col_cluster (bool): whether to cluster columns. If True, `row_cluster` must be False.
        row_cluster (bool): whether to cluster rows. If True, `col_cluster` must be False.

    Returns:
        2d array: clustered matrix
    """

    if col_cluster and row_cluster or not col_cluster and not row_cluster:
        raise ValueError("Exactly one of col_cluster and row_cluster must be True.")

    clustergrid = sns.clustermap(
        mtx,
        method="average",
        col_cluster=col_cluster,
        row_cluster=row_cluster,
        metric=metric,
    )
    plt.close()

    if col_cluster:
        idc = clustergrid.dendrogram_col.reordered_ind
    else:
        idc = clustergrid.dendrogram_row.reordered_ind

    conf_mtx_sorted = mtx[idc]
    conf_mtx_sorted = conf_mtx_sorted[:, idc]

    return conf_mtx_sorted, idc


def norm_conf_mtx(
    conf_mtx: np.ndarray, norm_axis: Optional[Union[str, Iterable[str]]] = None
):
    """
    Normalize a confusion matrix based on the specified normalization axis.

    Args:
        conf_mtx (np.ndarray): The confusion matrix to be normalized.
        norm_axis (Optional[Union[str, Iterable[str]]]): The axis or axes along which the normalization is performed.
            If None, no normalization is applied. If a string, it can be one of the following:
                - 'row': Normalize each row of the confusion matrix.
                - 'column': Normalize each column of the confusion matrix.
                - 'row_diag': Normalize the diagonal elements of each row of the confusion matrix.
                - 'column_diag': Normalize the diagonal elements of each column of the confusion matrix.
                - 'row_l1': Normalize each row of the confusion matrix using L1 normalization.
                - 'col_l1': Normalize each column of the confusion matrix using L1 normalization.
            If an iterable of strings, the normalization is applied sequentially along each specified axis.

    Returns:
        np.ndarray: The normalized confusion matrix.

    Raises:
        ValueError: If an invalid axis is provided.

    """
    if norm_axis is None:
        return conf_mtx

    elif type(norm_axis) is str:
        norm_axis = [norm_axis]

    for norm_axis in norm_axis:
        if norm_axis == "row":
            divisor = conf_mtx.max(axis=1)[:, None]
        elif norm_axis == "column":
            divisor = conf_mtx.max(axis=0)[None, :]
        elif norm_axis == "column_diag":
            divisor = conf_mtx.diagonal()[None, :]
        elif norm_axis == "row_diag":
            divisor = conf_mtx.diagonal()[:, None]
        elif norm_axis == "row_l1":
            divisor = np.sum(np.abs(conf_mtx), axis=1, keepdims=True)
        elif norm_axis == "col_l1":
            divisor = np.sum(np.abs(conf_mtx), axis=0, keepdims=True)
        else:
            raise ValueError(
                f"Invalid axis: {norm_axis}. Must be 'row', 'column', 'row_diag', 'column_diag', 'row_l1', or 'col_l1'."
            )

        divisor_positive = np.maximum(divisor, 1e-8)
        conf_mtx = conf_mtx / divisor_positive

    return conf_mtx


def compute_conf_mtx(
    transl_model,
    canvas_size,
    clust_imgs,
    clust_assignments,
    norm_pred_ac_imgs,
    select_unit_idc=None,
    repeats_dataloader=None,
    stds=None,
    means=None,
    device="cuda",
):
    """Compute model prediction (logit or activity) confusion matrix.

    Args:
        transl_model: model with readout maps shifted to be centered for all units
        canvas_size:
        clust_images: cluster images to compute confusion matrix for. Shape (n_clusters=1, n_rotations, h, w)
        norm_pred_ac_imgs: str, see allowed values for `type` in `norm_unit_act_imgs`
        select_unit_idc: indices of units to be considered
        repeats_dataloader: repeats_dataloader
        stds, means: for all units (not just those selected by `select_unit_idc`)

    Returns:
        NDArray: confusion matrix, shape (cluster_images, clusters)
    """

    warnings.warn("Currently only tested for max one min others controversial images")

    # present all rotations of cluster MEI to all neurons
    if clust_imgs[0].shape[0] > 1:  # (n_clusters, n_rotations, h, w)
        raise NotImplementedError("Only batch size 1 supported for cluster images.")

    preds = pred_max_rot(transl_model, canvas_size, clust_imgs, device)

    # # sanity check - long compute time
    # for clust_idx in range(len(preds)):
    #     print(clust_idx)  # TODO TEMP
    #     tab1 = (CITorchTransfMaxOneSuppOther().Unit() & sample_key & dict(cluster1_idx=clust_idx))
    #     tab2 = (ClusterAssignment().Unit() & dict(cluster_idx=clust_idx))

    #     assert np.allclose(
    #         preds[clust_idx][unit_idc == clust_idx],
    #         np.concatenate(np.array(
    #             (
    #                 tab1 * tab2
    #             ).fetch("prediction")
    #         )),
    #     ), f"Failed for cluster index {clust_idx}"

    preds = norm_unit_act_imgs(
        preds, norm_pred_ac_imgs, repeats_dataloader, stds, means
    )

    if select_unit_idc is not None:
        # flat_select_unit_idc = np.concatenate(select_unit_idc).flatten()
        preds = preds[:, select_unit_idc]
        clust_assignments = clust_assignments[select_unit_idc]

    # aggregate within cluster
    conf_mtx = [
        [
            pred_to_clust_img[clust_assignments == clust_idx].mean()
            for clust_idx in range(preds.shape[0])
        ]
        for pred_to_clust_img in preds
    ]  # (row: index of cluster for which cMEIs were computed, col: index of cluster to which the cMEIs were presented)
    conf_mtx = np.array(conf_mtx)

    return conf_mtx


def compute_pred_conf_mtx_raw(
    transl_model, canvas_size, clust_imgs, true_clust_id, select_unit_idc=None
):
    """Compute cluster prediction confusion matrix.

    Args:
        transl_model: model with readout maps shifted to be centered for all units
        canvas_size:
        clust_images: cluster images to compute confusion matrix for. Shape (n_clusters, n_batches=1, n_rotations, n_channels=1, h, w)
        true_clust_id: ground truth cluster id for all units (not just those selected by `select_idc`)
        select_idc: indices of units to be considered
    """
    with eval_state(transl_model):
        preds = pred_max_rot(transl_model, canvas_size, clust_imgs)

        if np.any(preds < 0):
            raise ValueError()

        pred_clust_id = np.argmax(preds, axis=0)

        if select_unit_idc is not None:
            pred_clust_id = pred_clust_id[select_unit_idc]
            true_clust_id = true_clust_id[select_unit_idc]

        pred_conf_mtx_raw = confusion_matrix(true_clust_id, pred_clust_id)
    return pred_conf_mtx_raw
