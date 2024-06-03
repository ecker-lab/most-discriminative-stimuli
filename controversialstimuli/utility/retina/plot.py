from typing import Any, List, Optional

from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
import numpy as np

from .constants import FRAME_RATE_MODEL

from .mei_analysis import decompose_kernel, calculate_fft, weighted_main_frequency


def plot_stimulus_composition(
    stimulus: np.array,
    temporal_trace_ax,
    freq_ax: Optional[Any],
    spatial_ax,
    lowpass_cutoff: float = 10.0,
    highlight_x_list: Optional[List[int]] = None,
):
    color_array = ["darkgreen", "darkviolet"]
    color_channel_names_array = ("Green", "UV")

    assert len(stimulus.shape) == 4
    num_color_channels, dim_t, dim_y, dim_x = stimulus.shape

    time_steps = stimulus.shape[1]
    stimulus_time = np.linspace(0, time_steps / FRAME_RATE_MODEL, time_steps)
    weighted_main_freqs = [0.0, 0.0]
    temporal_traces_max = 0.0
    temp_green, spat_green, _ = decompose_kernel(stimulus[0])
    temp_uv, spat_uv, _ = decompose_kernel(stimulus[1])
    temporal_kernels = [temp_green, temp_uv]

    # Spatial structure
    spatial_ax.set_title(f"Spatial Component {color_channel_names_array}")
    padding = np.zeros((spat_green.shape[0], 8))
    spat = np.concatenate([spat_green, padding, spat_uv], axis=1)

    abs_max = np.max([abs(spat.max()), abs(spat.min())])
    norm = Normalize(vmin=-abs_max, vmax=abs_max)
    spatial_ax.imshow(spat, cmap="RdBu_r", norm=norm)
    scale_bar = Rectangle(
        xy=(6, 15), width=3, height=1, color="k", transform=spatial_ax.transData
    )
    spatial_ax.annotate(
        text="150 µm",
        xy=(6, 14),
    )
    spatial_ax.add_patch(scale_bar)
    spatial_ax.axis("off")

    for color_idx in range(num_color_channels):
        temp = temporal_kernels[color_idx]
        temporal_traces_max = max(temporal_traces_max, np.abs(temp).max())

        # Temporal structure
        temporal_trace_ax.plot(stimulus_time, temp, color=color_array[color_idx])

        if freq_ax is not None:
            fft_freqs, fft_weights = calculate_fft(
                temp, FRAME_RATE_MODEL, lowpass_cutoff
            )
            weighted_main_freqs[color_idx] = weighted_main_frequency(
                fft_freqs, fft_weights
            )
            freq_ax.plot(fft_freqs, fft_weights, color=color_array[color_idx])

    temporal_trace_ax.set_ylim(-temporal_traces_max, +temporal_traces_max + 1)
    temporal_trace_ax.set_title("Temporal Trace of the Stimulus")
    temporal_trace_ax.set_xlabel("Time [s]")

    if freq_ax is not None:
        freq_ax.set_xlim(0.0, lowpass_cutoff + 1)
        freq_ax.set_xlabel("Frequency [Hz]")
        freq_ax.set_title(
            f"Weighted Frequency: {weighted_main_freqs[0]:.1f}/{weighted_main_freqs[1]:.1f} Hz"
            f" ({color_channel_names_array[0]}/{color_channel_names_array[1]})"
        )

    if highlight_x_list is not None:
        for x_0_idx, x_1_idx in highlight_x_list:
            x_0 = stimulus_time[x_0_idx]
            x_1 = stimulus_time[x_1_idx]
            temporal_trace_ax.fill_betweenx(
                temporal_trace_ax.get_ylim(), x_0, x_1, color="k", alpha=0.1
            )
