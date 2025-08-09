from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from sglang.multi_model.scheduling.stats import ScheduleStats

# Define a color map to assign different colors to each model
colors = ["blue", "green", "red", "purple", "orange", "brown", "pink"]


# Function to plot requests per second in separate subplots with different colors for each model
def plot_requests_per_second_in_bins(requests, trace_config=None, bin_size=1):
    # Dictionary to store the number of requests per second per model
    model_request_counts = defaultdict(list)

    # Find the minimum and maximum arrival times
    min_time = min(req.arrival_time for req in requests)
    max_time = max(req.arrival_time for req in requests)

    # Define the time bins based on the bin_size (e.g., 1-second bins)
    bins = np.arange(min_time, max_time + bin_size, bin_size)

    for req in requests:
        model = req.model
        model_request_counts[model].append(req.arrival_time)

    num_models = len(model_request_counts)

    # sort model_request_counts by descending order of the number of total requests
    model_request_counts = dict(sorted(model_request_counts.items()))

    # Create subplots with one plot for each model
    fig, axs = plt.subplots(num_models, 1, figsize=(12, 2 * num_models), sharex=True)

    if num_models == 1:
        axs = [axs]  # To handle the case when there's only one model

    # Determine the maximum y-axis limit across all subplots
    max_y = 0
    for arrival_times in model_request_counts.values():
        # Calculate histogram data without plotting
        counts, _ = np.histogram(arrival_times, bins=bins)
        max_y = max(max_y, counts.max())

    # Plot each model's request distribution in a separate axis
    for i, (model, arrival_times) in enumerate(model_request_counts.items()):
        color = colors[
            i % len(colors)
        ]  # Cycle through colors if there are more models than colors
        axs[i].hist(arrival_times, bins=bins, color=color)
        axs[i].set_title(f"Number of Requests per Second for {model}")
        axs[i].set_ylabel("Number of Requests")
        axs[i].set_ylim(0, max_y)  # Set the same y-axis limit for each subplot
        axs[i].grid(True)

    # Set the x-axis label only on the last subplot
    axs[-1].set_xlabel("Time (seconds)")

    if trace_config is not None:
        # set gobal figure title with alpha, req_rate, off_ratio
        plt.suptitle(
            f"Request Distribution (alpha={trace_config.alpha}, "
            f"req_rate={trace_config.req_rate}, off_ratio={trace_config.off_ratio}, "
            f"on_off_model_percentage={trace_config.on_off_model_percentage})",
            fontsize=16,
        )

    else:
        plt.suptitle("Request Distribution", fontsize=16)
    plt.tight_layout()
    plt.savefig("test1.pdf")


@dataclass
class PlotStats:
    time_idx: List[int]
    mod_req_per_sec: Dict[str, List[int]]
    mod_on_gpu: Dict[str, List[int]]
    mod_priority: Dict[str, List[float]]
    mod_q_waiting_len: Dict[str, List[int]]
    mod_req_vio: Optional[Dict[str, List[int]]] = None


def _convert_to_plotstats(schstats: List[ScheduleStats]) -> PlotStats:
    time_idx = list(range(len(schstats)))
    mod_req_per_sec = defaultdict(list)
    mod_on_gpu = defaultdict(list)
    mod_priority = defaultdict(list)
    mod_q_waiting_len = defaultdict(list)
    mod_req_vio = defaultdict(list)

    for stats in schstats:
        for model, req_num in stats.mod_req_per_sec.items():
            mod_req_per_sec[model].append(req_num)
        for model, gpu_id in stats.mod_on_gpu.items():
            mod_on_gpu[model].append(gpu_id)
        for model, priority in stats.mod_priority.items():
            mod_priority[model].append(priority)
        for model, q_waiting_len in stats.mod_q_waiting_len.items():
            mod_q_waiting_len[model].append(q_waiting_len)
        if stats.mod_req_vio is not None:
            for model, req_vio in stats.mod_req_vio.items():
                mod_req_vio[model].append(req_vio)

    return PlotStats(
        time_idx=time_idx,
        mod_req_per_sec=dict(mod_req_per_sec),
        mod_on_gpu=dict(mod_on_gpu),
        mod_priority=dict(mod_priority),
        mod_q_waiting_len=dict(mod_q_waiting_len),
        mod_req_vio=dict(mod_req_vio) if mod_req_vio else None,
    )


def plot_sim_stats(schstats: List[ScheduleStats], file_name, bin_size=1):

    plotstats = _convert_to_plotstats(schstats)

    # Dictionary to store the number of requests per second per model
    model_request_counts = defaultdict(list)

    # Find the minimum and maximum arrival times
    min_time = plotstats.time_idx[0]
    max_time = plotstats.time_idx[-1]

    # Define the time bins based on the bin_size (e.g., 1-second bins)
    bins = np.arange(min_time, max_time + bin_size, bin_size)
    model_name_to_idx = {
        model: idx for idx, model in enumerate(plotstats.mod_req_per_sec.keys())
    }

    num_models = len(plotstats.mod_req_per_sec.keys())

    # Compute the number of GPUs dynamically using a set
    gpu_ids = set()
    for time_points in plotstats.mod_on_gpu.values():
        gpu_ids.update(time_points)  # Add all GPU IDs (including -1 if present)
    gpu_ids.discard(-1)
    num_gpus = len(gpu_ids)

    # Sort model_request_counts by descending order of the number of total requests
    model_request_counts = dict(sorted(plotstats.mod_req_per_sec.items()))

    # Total subplots = num_models (request plots) + num_gpus (GPU availability) + 2 (queue waiting length and violations)
    if plotstats.mod_req_vio is not None:
        total_subplots = num_models + num_gpus + 2
    else:
        total_subplots = num_models + num_gpus + 1

    # Create subplots with one plot for each model, GPU, and additional plots
    fig, axs = plt.subplots(
        total_subplots, 1, figsize=(12, 2 * total_subplots), sharex=True
    )

    if total_subplots == 1:
        axs = [axs]  # To handle the case when there's only one subplot

    # Determine the maximum y-axis limit across all subplots
    max_y = 0
    for request_counts in plotstats.mod_req_per_sec.values():
        max_y = max(
            max_y, max(request_counts)
        )  # Update max_y based on the max request count for each model

    # Plot each model's request distribution in a separate axis
    for i, (model, request_counts) in enumerate(plotstats.mod_req_per_sec.items()):
        color = colors[
            model_name_to_idx[model] % len(colors)
        ]  # Cycle through colors if more models than colors
        axs[i].bar(
            range(len(request_counts)), request_counts, color=color, width=0.98
        )  # Plot a bar chart
        axs[i].set_title(f"Requests per Second for {model}")
        axs[i].set_ylabel("Number of Requests")
        axs[i].set_xlabel("Time (s)")
        axs[i].set_ylim(0, max_y)  # Set the same y-axis limit for each subplot
        axs[i].grid(True)

    """Model availability on GPUs
    """
    for gpu_id in range(num_gpus):
        axs_on_gpu = axs[num_models + gpu_id]
        axs_on_gpu.set_title(f"Model Availability on GPU {gpu_id}")
        axs_on_gpu.set_xlabel("Time")
        axs_on_gpu.set_ylabel("Model ID")
        axs_on_gpu.grid(True)

        for mod_name, time_points in plotstats.mod_on_gpu.items():
            for t, active in enumerate(time_points):
                if active == gpu_id:  # Model is on this GPU
                    axs_on_gpu.barh(
                        mod_name,
                        1,
                        left=t,
                        height=0.3,
                        color=colors[model_name_to_idx[mod_name] % len(colors)],
                    )
                    # Add priority text
                    if t % 5 == 0 and t < len(plotstats.mod_priority[mod_name]):
                        axs_on_gpu.text(
                            t + 0.5,
                            mod_name,
                            f"{plotstats.mod_priority[mod_name][t]:.2f}",
                            va="center",
                            ha="center",
                            fontweight="bold",
                            fontsize=6,
                            color="black",
                        )

    """Plot each model's queue waiting length as a line
    """
    axs_q_wait = axs[num_models + num_gpus]
    for mod_name, waiting_lengths in plotstats.mod_q_waiting_len.items():
        axs_q_wait.plot(
            waiting_lengths,
            label=f"Model {mod_name}",
            color=colors[model_name_to_idx[mod_name] % len(colors)],
        )
        axs_q_wait.grid(True)

    # Set labels and title
    axs_q_wait.set_ylabel("Queue Waiting Length")
    axs_q_wait.set_title("Queue Waiting Length Over Time for Each Model")

    """Plot each model's request violation as a line
    """
    if plotstats.mod_req_vio is not None:
        axs_q_vio = axs[num_models + num_gpus + 1]
        for mod_name, vio_num in plotstats.mod_req_vio.items():
            axs_q_vio.plot(
                vio_num,
                label=f"Model {mod_name}",
                color=colors[model_name_to_idx[mod_name] % len(colors)],
            )
            axs_q_vio.grid(True)

        # Set labels and title
        axs_q_vio.set_ylabel("Violation requests")
        axs_q_vio.set_title("Accumulated Request Violation for Each Model")

    # Set the x-axis label only on the last subplot
    axs[-1].set_xlabel("Time (seconds)")

    # Set global figure title with alpha, req_rate, off_ratio
    plt.suptitle(
        f"Model availability",
        fontsize=16,
    )

    plt.tight_layout()
    plt.savefig(file_name)
