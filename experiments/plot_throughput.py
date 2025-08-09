import argparse
import re

import matplotlib.pyplot as plt
import pandas as pd

model_colors = {
    "model_1": "blue",
    "model_2": "green",
    "model_3": "red",
    "model_4": "purple",
    "model_5": "orange",
    "model_6": "brown",
    "model_7": "pink",
    "model_8": "gray",
    "model_9": "olive",
    "model_10": "cyan",
    "model_11": "magenta",
    "model_12": "teal",
    "model_13": "maroon",
    "model_14": "turquoise",
    "model_15": "indigo",
    "model_16": "lime",
    "model_17": "lavender",
    "model_18": "mint",
}


def analyze_log(file_name, start_time, end_time, output_file):
    # Define the regex pattern for extracting data.
    throughput_pattern = (
        r"\[(.*?)\s(\w+_\d+)\sGPU=(\d+)\s.*?gen throughput \(token/s\): (\d+\.\d+)"
    )
    activate_start_pattern = (
        r"\[(.*?)\s(\w+_\d+)\sGPU=(\d+)\s.*?Scheduler receives the activate request"
    )
    activate_end_pattern = r"\[(.*?)\s(\w+_\d+)\sGPU=(\d+)\s.*?Scheduler activated"
    deactivate_start_pattern = (
        r"\[(.*?)\s(\w+_\d+)\sGPU=(\d+)\s.*?Scheduler receives the deactivate requests"
    )
    deactivate_end_pattern = (
        r"\[(.*?)\s(\w+_\d+)\sGPU=(\d+)\s.*?Total time taken for deactivation"
    )

    # Read and parse the log file.
    data = []
    activate_start_events = []
    activate_end_events = []
    deactivate_start_events = []
    deactivate_end_events = []
    with open(file_name, "r") as file:
        for line in file:
            match = re.search(throughput_pattern, line)
            if match:
                timestamp, model_name, gpu_id, throughput = match.groups()
                data.append(
                    {
                        "timestamp": timestamp,
                        "model_name": model_name,
                        "gpu_id": gpu_id,
                        "throughput": float(throughput),
                    }
                )
            match = re.search(activate_start_pattern, line)
            if match:
                activate_start_events.append(match.groups())
            match = re.search(activate_end_pattern, line)
            if match:
                activate_end_events.append(match.groups())
            match = re.search(deactivate_start_pattern, line)
            if match:
                deactivate_start_events.append(match.groups())
            match = re.search(deactivate_end_pattern, line)
            if match:
                timestamp, model_name, gpu_id = match.groups()
                deactivate_end_events.append((timestamp, model_name, gpu_id))
                data.append(
                    {
                        "timestamp": timestamp,
                        "model_name": model_name,
                        "gpu_id": gpu_id,
                        "throughput": 0,
                    }
                )

    # Create a DataFrame from the parsed data.
    df = pd.DataFrame(data)

    # Convert the timestamp column to datetime.
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Filter data based on the provided time range.
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)
    filtered_df = df[(df["timestamp"] >= start_time) & (df["timestamp"] <= end_time)]
    num_gpus = filtered_df["gpu_id"].nunique()
    fig, axs = plt.subplots(num_gpus * 2, 1, figsize=(18, 6 * num_gpus))
    # Group by GPU ID and plot for each GPU separately.
    for gpu_id, gpu_group in filtered_df.groupby("gpu_id"):
        for model_name, group in gpu_group.groupby("model_name"):
            group = group.sort_values("timestamp")
            idx = gpu_id * 2
            axs[idx].plot(
                group["timestamp"],
                group["throughput"],
                label=model_name,
                marker="o",
                color=model_colors.get(model_name, "black"),
            )

        count = 0
        for start_event in activate_start_events:
            timestamp, model_name, event_gpu_id = start_event
            if event_gpu_id == gpu_id:
                timestamp = pd.to_datetime(timestamp)
                if timestamp >= start_time and timestamp <= end_time:
                    label = f"Activate Start: {model_name}" if count == 0 else None
                    axs[idx].axvline(
                        x=timestamp,
                        color=model_colors.get(model_name, "black"),
                        linestyle="solid",
                        label=label,
                    )
                    count += 1
        count = 0
        for end_event in activate_end_events:
            timestamp, model_name, event_gpu_id = end_event
            if event_gpu_id == gpu_id:
                timestamp = pd.to_datetime(timestamp)
                if timestamp >= start_time and timestamp <= end_time:
                    label = f"Activate End: {model_name}" if count == 0 else None
                    axs[idx].axvline(
                        x=timestamp,
                        color=model_colors.get(model_name, "black"),
                        linestyle="--",
                        label=label,
                    )
                    count += 1

        # second figure: deactivate events
        for model_name, group in gpu_group.groupby("model_name"):
            group = group.sort_values("timestamp")
            axs[idx + 1].plot(
                group["timestamp"],
                group["throughput"],
                label=model_name,
                marker="o",
                color=model_colors.get(model_name, "black"),
            )

        count = 0
        for start_event in deactivate_start_events:
            timestamp, model_name, event_gpu_id = start_event
            if event_gpu_id == gpu_id:
                timestamp = pd.to_datetime(timestamp)
                if timestamp >= start_time and timestamp <= end_time:
                    label = f"Deactivate Start: {model_name}" if count == 0 else None
                    axs[idx + 1].axvline(
                        x=timestamp,
                        color=model_colors.get(model_name, "black"),
                        linestyle="solid",
                        label=label,
                    )
                    count += 1

        count = 0
        for end_event in deactivate_end_events:
            timestamp, model_name, event_gpu_id = end_event
            if event_gpu_id == gpu_id:
                timestamp = pd.to_datetime(timestamp)
                if timestamp >= start_time and timestamp <= end_time:
                    label = f"Deactivate End: {model_name}" if count == 0 else None
                    axs[idx + 1].axvline(
                        x=timestamp,
                        color=model_colors.get(model_name, "black"),
                        linestyle="--",
                        label=label,
                    )
                    count += 1

        # subtitles
        axs[0].set_title(f"Throughput and Activate Events for GPU {gpu_id}")
        axs[1].set_title(f"Throughput and Deactivate Events for GPU {gpu_id}")

        plt.xlabel("Timestamp")
        plt.ylabel("Throughput (tokens/s)")
        plt.xticks(rotation=45)
        plt.legend(title="Model Name")
        plt.grid()
        plt.tight_layout()

        plt.savefig(f"{output_file}_gpu_{gpu_id}.png")
        print(f"Plot saved as {output_file}_gpu_{gpu_id}.png")


if __name__ == "__main__":
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(
        description="Analyze log file and plot throughput."
    )
    parser.add_argument("--log", type=str, help="Path to the log file.")
    parser.add_argument(
        "--start",
        type=str,
        help="Start time for filtering (format: 'YYYY-MM-DD HH:MM:SS').",
    )
    parser.add_argument(
        "--end",
        type=str,
        help="End time for filtering (format: 'YYYY-MM-DD HH:MM:SS').",
    )
    parser.add_argument("--output", type=str, help="Path to the output file.")

    args = parser.parse_args()
    analyze_log(args.log, args.start, args.end, args.output)

# python plot_throughput.py --log server-logs/one_queue_120302.log --start "2024-12-04 01:27:29" --end "2024-12-04 01:41:00" --output ./throughput_one_queue120302.png
