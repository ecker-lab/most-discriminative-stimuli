from typing import Any, Dict, Optional, Tuple, List
from abc import ABC, abstractmethod

from datajoint import table
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from math import ceil
import seaborn as sns
from numpy.typing import ArrayLike
import scipy

from controversialstimuli.utility.retina.constants import RGC_GROUP_NAMES_DICT_SHORT_PAPER


def config_seaborn(
    context: str = "paper",
    style: str = "ticks",
    palette: str = "viridis",  # "deep"
    font: str = "sans-serif",
    font_scale: int = 1,
    color_codes: bool = True,
    rc: Dict[str, Any] = None,
) -> None:
    sns.set_theme(context, style, palette, font, font_scale, color_codes, rc)


def imshow_gray_sym(img, ax=None, axis="off", vmax_abs=None, mark_center=False):
    ax = ax or plt.subplots()[1]
    vmax = np.max(np.abs(img))
    vmax = vmax_abs or np.max(np.abs(img))
    if mark_center:
        ax.scatter([img.shape[1] / 2], [img.shape[0] / 2], color="black", marker="x", s=1)
    ret = ax.imshow(img.squeeze(), cmap="gray", vmin=-vmax, vmax=vmax)
    ax.axis(axis)
    return ret


def get_subplots_grid(num_imgs, *args, ncols=6, size_scale=2, axes_off=True, **kwargs):
    """Generate grid of subplots.

    Args:
        num_panels (int): Number of images that should be plotted
        ncols (int, optional): Number of columns. Defaults to 6.
        size_scale (int, optional): Scale the size of the figure, figsize=(size_scale * ncols, size_scale * nrows). Defaults to 2.
        *args, **kwargs: Anything that `plt.subplots(*args, **kwargs)` takes as argument.

    Returns:
        Tuple: figure, axes
    """
    nrows = ceil(num_imgs / ncols)
    fig, axes = plt.subplots(nrows, ncols, *args, figsize=(size_scale * ncols, size_scale * nrows), **kwargs)

    if axes_off:
        for ax in axes.flatten():
            ax.axis("off")

    return fig, axes


def normalize_image_unit_interval(img: ArrayLike) -> np.ndarray:
    """Normalize image to be symetrically between -1 and 1 without bias (non-affine).
        I.e. -1 xor 1 do not necessarily have to be matched.

    Args:
        img (np.ndarray): Image to be normalized.
        min_val (float, optional): Minimum value. Defaults to -1.
        max_val (float, optional): Maximum value. Defaults to 1.

    Returns:
        np.ndarray: Normalized image.
    """
    img = np.array(img)
    extremum = np.max(np.abs(img))
    return img / extremum


def cm2inch(x: float) -> float:
    return x / 2.54


def inch2cm(x: float) -> float:
    return x * 2.54


def image_mid(img, dim):
    return (img.shape[dim] + 1) // 2


def show_grid(ax, xpos, ypos):
    ax.set_xticks(xpos)
    ax.set_xticklabels([None])
    ax.set_yticks(ypos)
    ax.set_yticklabels([None])
    # ax.grid(color='r', linestyle='-', alpha=0.2, linewidth=1)
    # ax.grid(color=(0.5, 0.45, 0.45), linestyle='-', linewidth=1)
    ax.grid(color=(147 / 255, 109 / 255, 109 / 255), linestyle="-", linewidth=1)


def plot_mei(ax, key, restriction, mei_tab, title_clust_id, title_prefix, show_metrics=True):
    img = (mei_tab() & key & restriction).get_images(field_name="cluster_images").squeeze()[0]
    imshow_gray_sym(img, ax=ax, axis="on")
    show_grid(ax, [image_mid(img, 1)], [image_mid(img, 0)])
    avg_pred = (mei_tab() & key & restriction).fetch1("avg_prediction")

    title = title_prefix
    if show_metrics:
        title += "\ncluster avg. pred: " + np.round(avg_pred, 2).astype(str)
    ax.set_title(title)


def plot_ci(ax, key, restriction, ci_tab, title_prefix, show_metrics=True):
    img = (ci_tab() & key & restriction).get_images(field_name="cluster_images").squeeze()[0]
    imshow_gray_sym(img, ax=ax, axis="on")
    show_grid(ax, [image_mid(img, 1)], [image_mid(img, 0)])

    avg_pred_drive = (ci_tab() & key & restriction).fetch1("avg_pred_drive")
    avg_pred_suppress = (ci_tab() & key & restriction).fetch1("avg_pred_suppress")

    title = title_prefix
    if show_metrics:
        title += (
            ", cluster avg. pred drive: "
            + np.round(avg_pred_drive, 2).astype(str)
            + "\navg. pred suppress: "
            + np.round(avg_pred_suppress, 2).astype(str)
        )
    ax.set_title(title)


def plot_conf_mtx(conf_mtx, annot, xticklabels="auto", yticklabels="auto", vmin=None, vmax=None, center=None, **kwargs):
    """Plot confusion matrix.

    Args:
        conf_mtx (np.ndarray): Confusion matrix.
        center (float, optional): Value at which to center colormap to plot diverging data. Defaults to None."""

    fig, ax = plt.subplots(figsize=(cm2inch(0.5) * len(conf_mtx), cm2inch(0.5) * len(conf_mtx)))

    sns.heatmap(
        data=conf_mtx,
        annot=annot,
        fmt="",
        ax=ax,
        square=True,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        vmin=vmin,
        vmax=vmax,
        center=center,
        annot_kws={"fontsize": 6},
        cbar_kws={"shrink": 0.8},
        **kwargs,
    )  # , annot_kws={"fontsize":30})
    ax.set_ylabel("Presented cluster image")
    ax.set_xlabel("Predicted cluster mean response")

    return fig


class ConfMtxLabelPlotter(ABC):
    @abstractmethod
    def plot(self, axes: List) -> None:
        """Plots the the single MDS or multiple images of a SVD decomposed MDS onto the axes."""
        pass

    def num_of_required_axes(self) -> int:
        """Specifies how many aces are required to plot the MDS / MDS decomposition."""
        return 1


class ImagePlotter(ConfMtxLabelPlotter):
    def __init__(self, image: np.array):
        """
        Args:
            image: Image to be plotted.
        """
        super().__init__()
        self._image = image

    def plot(self, axes: List):
        assert len(axes) == self.num_of_required_axes()
        imshow_gray_sym(self._image, axes[0])


def decompose_kernel(space_time_kernel: np.array, scaling_factor: float = 1.0) -> Tuple[np.array, np.array, np.array]:
    """
    Computes the SVD of a space-time kernel. Scales spatial and temporal components such that spatial component is in 
        the range [-1, 1]
    Args:
        space_time_kernel: shape=[time, y_shape, x.shape]
        scaling_factor: optional scaling factor of the temporal kernel.
    Returns: temporal kernel, reshaped spatial kernel, singular values
    """
    assert len(space_time_kernel.shape) == 3
    dt, dy, dx = space_time_kernel.shape

    space_time_kernel_flat = space_time_kernel.reshape((dt, dy * dx))
    U, S, V = scipy.linalg.svd(space_time_kernel_flat)
    temporal = U[:, 0]
    spatial = V[0]
    scaling_factor *= S[0]
    abs_max_val = max(np.abs(spatial.min()), np.abs(spatial.max()))
    spatial = spatial / abs_max_val
    temporal = temporal * abs_max_val * scaling_factor
    reshaped_spatial = spatial.reshape((dy, dx))

    # Flip sign if center of spatial component is negative
    center_y, center_x = int(dy / 2), int(dx / 2)
    spatial_center = reshaped_spatial[center_y - 1 : center_y + 2, center_x - 1 : center_x + 2]
    spatial_center_mean = np.mean(spatial_center)
    if spatial_center_mean < 0:
        spatial *= -1
        temporal *= -1
    singular_values = S[1]
    return temporal, reshaped_spatial, singular_values


class MoviePlotter(ConfMtxLabelPlotter):
    def __init__(self, movie: np.array, highlight_region: Tuple[int, int] = (40, 50)):
        """
        Args:
            movie: Movie to be plotted. Shape (2 (green, uv color channel), time, h, w)
            highlight_region: Region to be highlighted in the temporal plot.
        """
        super().__init__()
        assert len(movie.shape) == 4
        self._temporal_green, self._spatial_green, _ = decompose_kernel(movie[0])
        self._temporal_uv, self._spatial_uv, _ = decompose_kernel(movie[1])
        self._highlight_region = highlight_region

    def abs_max_uv(self) -> float:
        return np.abs(self._temporal_uv).max()

    def abs_max_green(self) -> float:
        return np.abs(self._temporal_green).max()

    def norm_uv(self, max_value: float):
        self._temporal_uv /= max_value

    def norm_green(self, max_value: float):
        self._temporal_green /= max_value

    def plot(self, axes: List, flip_axes: bool = False) -> None:
        """
        Args:
            flip_axes: If True, the axes plotting order is flipped. Default: False, then the plotting order is spatial
                green, spatial uv, temporal.
        """
        green_color = "darkgreen"
        uv_color = "darkviolet"

        assert len(axes) == self.num_of_required_axes()
        if flip_axes:
            axes = axes[::-1]
        spatial_ax_green, spatial_axes_uv, temporal_ax = axes
        temporal_ax.plot(self._temporal_green, color=green_color)
        temporal_ax.plot(self._temporal_uv, color=uv_color)
        temporal_ax.fill_betweenx(
            temporal_ax.get_ylim(), self._highlight_region[0], self._highlight_region[1], color="k", alpha=0.1
        )

        spatial_plot_list = [
            (spatial_ax_green, self._spatial_green, green_color),
            (spatial_axes_uv, self._spatial_uv, uv_color),
        ]
        for spatial_ax, spatial_comp, color in spatial_plot_list:
            abs_max = np.max([abs(spatial_comp.max()), abs(spatial_comp.min())])
            norm = Normalize(vmin=-abs_max, vmax=abs_max)
            spatial_ax.imshow(spatial_comp, cmap="RdBu_r", norm=norm)
            # Add a colored box around the image
            spatial_ax.add_patch(Rectangle((0, 0), spatial_comp.shape[1] - 1, spatial_comp.shape[0] - 1,
                                           edgecolor=color, facecolor='none', linewidth=2))

        for ax in axes:
            ax.set_axis_off()

    def num_of_required_axes(self) -> int:
        return 3


def plot_conf_mtx_imgs(
    mtx: np.ndarray,
    annot,
    h_plotter_list: List[ConfMtxLabelPlotter],
    v_plotter_list: Optional[List[ConfMtxLabelPlotter]],
    fmt="",
    title="",
    xticklabels="auto",
    yticklabels="auto",
    xlabel=None,
    ylabel=None,
    vmin=None,
    vmax=None,
    center=None,
    cbar=True,
    figsize_scale=0.6,
    figsize=None,
    annot_kws=None,
    paper=False,
    **kwargs,
) -> plt.Figure:
    """
    Plot a heatmap of a confusion matrix with additional image plots as the labels.

    Args:
        mtx (np.ndarray): The confusion matrix.
        annot: Data used to annotate the heatmap cells. See sns.heatmap for more information.
        h_plotter_list (List[ConfMtxLabelPlotter]): List of horizontal label plotters.
        v_plotter_list (Optional[List[ConfMtxLabelPlotter]]): List of vertical label plotters.
        fmt (str, optional): String formatting code to use when adding annotations. See sns.heatmap for details. 
            Defaults to "".
        title (str, optional): Title of the plot. Defaults to "".
        xticklabels (str or list, optional): Labels for the x-axis ticks. See sns.heatmap for details. 
            Defaults to "auto".
        yticklabels (str or list, optional): Labels for the y-axis ticks. See sns.heatmap for details. 
            Defaults to "auto".
        xlabel (str, optional): Label for the x-axis. Defaults to None.
        ylabel (str, optional): Label for the y-axis. Defaults to None.
        vmin (float, optional): Minimum value of the colorbar. Defaults to None.
        vmax (float, optional): Maximum value of the colorbar. Defaults to None.
        center (float, optional): Value at which to center the colorbar. Defaults to None.
        cbar (bool, optional): Whether to show the colorbar. Defaults to True.
        figsize_scale (float, optional): Scaling factor for the figure size. Defaults to 0.6.
        figsize (tuple, optional): Figure size in inches. Defaults to None.
        annot_kws (dict, optional): Additional keyword arguments for annotating the heatmap. See sns.heatmap for 
            details. Defaults to None.
        paper (bool, optional): Whether to optimize the plot for printing on paper. Defaults to False.
        **kwargs: Additional keyword arguments for the seaborn heatmap.

    Returns:
        plt.Figure: The generated figure.
    """
    if annot_kws is None:
        annot_kws = {"fontsize": 7}

    num_h_plotting_axes = h_plotter_list[0].num_of_required_axes()
    num_v_plotting_axes = num_h_plotting_axes if v_plotter_list is None else v_plotter_list[0].num_of_required_axes()
    num = len(h_plotter_list)
    subplot_shape = (num + num_v_plotting_axes, num + num_h_plotting_axes)
    if figsize:
        print('overwrite figsize_scale argument')
    figsize = figsize or cm2inch(figsize_scale) * np.array(subplot_shape)
    fig = plt.figure(figsize=figsize)

    v_axes = [
        [plt.subplot2grid(subplot_shape, (i, subplot_shape[1] - k)) for k in range(1, num_v_plotting_axes + 1)]
        for i in range(num_h_plotting_axes, subplot_shape[0])
    ]
    h_axes = [
        [plt.subplot2grid(subplot_shape, (k, i)) for k in range(0, num_h_plotting_axes)]
        for i in range(0, subplot_shape[1] - num_h_plotting_axes)
    ]
    main_ax = plt.subplot2grid(
        subplot_shape,
        (num_v_plotting_axes, 0),
        rowspan=subplot_shape[0] - num_v_plotting_axes,
        colspan=subplot_shape[1] - num_h_plotting_axes,
    )

    for plotter, ax_list in zip(v_plotter_list if v_plotter_list is not None else h_plotter_list, v_axes):
        plotter.plot(ax_list)

    for plotter, ax_list in zip(h_plotter_list, h_axes):
        plotter.plot(ax_list)

    sns.heatmap(
        data=mtx,
        annot=annot,
        fmt=fmt,
        ax=main_ax,
        square=True,
        cbar=cbar,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        vmin=vmin,
        vmax=vmax,
        center=center,
        annot_kws=annot_kws,
        cbar_kws={"shrink": 0.4} if not paper else {},
        **kwargs,
    )  # , annot_kws={"fontsize":30})
    main_ax.set_yticklabels(main_ax.get_yticklabels(), rotation=0)

    main_ax.set_ylabel("Most discriminative stimulus" if ylabel is None else ylabel)
    main_ax.set_xlabel("Cluster mean response" if xlabel is None else xlabel)
    if len(title) > 0:
        fig.suptitle(title)
    plt.subplots_adjust(left=0.175, right=0.84, bottom=0.17, top=0.83, wspace=0, hspace=0)

    return fig



def get_main_diagonal_annotation(conf_mtx: np.ndarray) -> np.ndarray:
    """Returns an annotation array (all elements are strings) only annotating the main diagonal of the confusion matrix."""
    annot = np.zeros_like(conf_mtx)
    np.fill_diagonal(annot, conf_mtx.diagonal())
    annot = np.round(annot, 1)
    annot = annot.astype("str")
    annot[annot == "0.0"] = ""
    return annot


def plot_responses(
        list_of_responses: List[np.array],
        ax,
        num_individual_responses_to_plot: int = 10,
        label: Optional[str] = None,
        highlight_x_list: Optional[List[int]] = None,
        time_offset_in_seconds: float = 1.0,
        color: Optional[str] = None,
        y_max: Optional[float] = None,
):
    mean = np.mean(list_of_responses, axis=0)
    FRAME_RATE_MODEL = 30.0
    time = (np.arange(mean.shape[0]) / FRAME_RATE_MODEL) + time_offset_in_seconds
    stddev = np.std(list_of_responses, axis=0)
    for i, response in enumerate(list_of_responses):
        if i >= num_individual_responses_to_plot:
            break
        pass

    ax.plot(time, mean, label=label, color=color)

    if highlight_x_list is not None:
        for x_0_idx, x_1_idx in highlight_x_list:
            x_0 = time[x_0_idx]
            x_1 = time[x_1_idx]
            ylim = (0.0, y_max) if y_max is not None else ax.get_ylim()
            ax.fill_betweenx(ylim, x_0, x_1, color="k", alpha=0.1)


def plot_traces(
    traces: Dict[int, List[np.array]],
    cei_plotter_list: List[MoviePlotter],
    mei_plotter_list: List[MoviePlotter],
    figsize_scale=8.0,
):
    num_h_plotting_axes = cei_plotter_list[0].num_of_required_axes()
    num_mei_plotting_axes = mei_plotter_list[0].num_of_required_axes()

    rgc_ids_set_list = [set(v.keys()) for v in traces.values()]
    all_rgc_set = set()
    for single_rgc_set in rgc_ids_set_list:
        all_rgc_set = all_rgc_set.union(single_rgc_set)
    # color_palatte ignores the n_colors arg if as_cmap is True
    # Therefore, we divide the colorspace ourselves
    cmap = sns.color_palette("husl", as_cmap=True)
    rgc_id_to_cmap_id = {
        rgc_id: int(i * cmap.N / len(all_rgc_set))
        for i, rgc_id in enumerate(all_rgc_set)
    }
    rgc_to_color_dict = {rgc_id: cmap(cmap_id) for rgc_id, cmap_id in rgc_id_to_cmap_id.items()}

    num_neurons_per_cluster = {}
    max_value_traces = 0.0
    for c_id, traces_list in traces.items():
        if type(traces_list) is list:
            num_neurons_per_cluster[c_id] = len(traces_list)
            max_val = max(x.max() for x in traces_list)
            max_value_traces = max(max_val, max_value_traces)
        elif type(traces_list) is dict:
            num = sum(len(x) for x in traces_list.values())
            num_neurons_per_cluster[c_id] = num
            for t in traces_list.values():
                max_val_of_mean_traces = np.mean(t, axis=0).max()
                max_value_traces = max(max_val_of_mean_traces, max_value_traces)
    cluster_ids = sorted(traces.keys())

    num = len(cluster_ids)
    fig_rows = num + num_mei_plotting_axes
    fig_cols = num + num_h_plotting_axes
    plot_shape = (fig_rows, fig_cols)
    figsize = cm2inch(figsize_scale) * np.array(plot_shape)
    plt.rcParams.update({'font.size': 30})
    fig, axes = plt.subplots(fig_rows, fig_cols, figsize=figsize)

    for i, c_id in enumerate(cluster_ids):
        plotter = cei_plotter_list[c_id]
        axes_for_plots = axes[i+num_mei_plotting_axes, :num_h_plotting_axes]
        plotter.plot(axes_for_plots)
        axes_for_mei_plots = axes[:num_mei_plotting_axes, i+num_h_plotting_axes]
        mei_plotter = mei_plotter_list[c_id]
        mei_plotter.plot(axes_for_mei_plots, flip_axes=True)
    for col_id in range(num_h_plotting_axes):
        for row_id in range(num_mei_plotting_axes):
            axes[col_id][row_id].set_axis_off()

    plotting_axes = axes[num_mei_plotting_axes:, num_h_plotting_axes:]
    for cluster_pos, cluster_id in enumerate(cluster_ids):
        trace_list = traces[cluster_id]
        for stim_pos, stim_id in enumerate(cluster_ids):
            curr_ax = plotting_axes[stim_pos, cluster_pos]
            highlight_x_list = [(10, 19)]
            if type(trace_list) is list:
                trace_list_stim = [trace[stim_id] for trace in trace_list]
                plot_responses(trace_list_stim, ax=curr_ax, highlight_x_list=highlight_x_list, color="blue")
            elif type(trace_list) is dict:
                rgc_types_sorted = sorted(list(trace_list.keys()), key=lambda idx: -len(trace_list[idx]))
                for rgc_it, rgc_type in enumerate(rgc_types_sorted):
                    rgc_trace_list = trace_list[rgc_type]
                    trace_list_stim = [trace[stim_id] for trace in rgc_trace_list]
                    color = rgc_to_color_dict[rgc_type]
                    plot_responses(trace_list_stim, ax=curr_ax, highlight_x_list=highlight_x_list,
                                   label=rgc_type, y_max=max_value_traces, color=color)
                    highlight_x_list = None
                # curr_ax.legend(loc="upper left")
            else:
                assert False
            eps_y_lim = 0.0
            curr_ax.set_ylim([-eps_y_lim, max_value_traces + eps_y_lim])
            curr_ax.set_axis_on()
            curr_ax.set_xticks([])
            curr_ax.set_yticks([])

            if (stim_pos + 1) == plotting_axes.shape[0]:
                num_neurons = num_neurons_per_cluster[cluster_id]
                curr_ax.set_xlabel(f"{cluster_id+1} [{num_neurons}]")
            if cluster_pos == 0:
                curr_ax.set_ylabel(f"Stim. {stim_id+1}")

    # Add xlabel and ylabel descriptions
    # For font size see: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.text.html
    # x-axis 
    fig.text(0.6, 0.08, 'Cluster [# neurons]', ha='center')
    # x-axis top
    fig.text(0.58, 0.9, "Each cluster's group MEI", va='center')
    # y-axis
    fig.text(0.1, 0.4, 'Presented maximally discriminative stimulus', va='center', rotation='vertical')

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    labels_lines_dict = {la: li for la, li in zip(labels, lines)}
    sorted_labels = sorted(labels_lines_dict.keys(), key=lambda x : int(x))
    lines_list = [labels_lines_dict[k] for k in sorted_labels]
    sorted_labels_names = [RGC_GROUP_NAMES_DICT_SHORT_PAPER[int(x)] for x in sorted_labels]
    fig.legend(lines_list, sorted_labels_names, loc='lower left', bbox_to_anchor=(0.9, 0.2))

    return fig

