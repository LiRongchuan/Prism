import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def plot_gpu_and_memory_usage(csv_file):
    # 读取 CSV
    df = pd.read_csv(csv_file)
    
    # 检查列名
    expected_columns = ["timestamp", "gpu_util_percent", "memory_used_MB", "memory_total_MB"]
    if list(df.columns) != expected_columns:
        raise ValueError(f"CSV columns mismatch. Expected: {expected_columns}, but got: {list(df.columns)}")
    
    # 时间戳 → 相对时间（秒）
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["time_sec"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
    
    # **关键**：内存对齐—减去初始已用内存，使得第一点为 0
    df["memory_used_offset_MB"] = df["memory_used_MB"] - df["memory_used_MB"].iloc[0]
    
    # 创建图表
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 左轴：GPU 利用率
    ax1.plot(df["time_sec"], df["gpu_util_percent"],
             color="tab:blue", label="GPU Utilization (%)")
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("GPU Utilization (%)", color="tab:blue")
    ax1.tick_params(axis='y', labelcolor="tab:blue")

    # 右轴：内存增量（对齐后）
    ax2 = ax1.twinx()
    ax2.plot(df["time_sec"], df["memory_used_offset_MB"],
             color="tab:red", label="Memory Used Δ (MB)")
    ax2.set_ylabel("Memory Used Increase (MB)", color="tab:red")
    ax2.tick_params(axis='y', labelcolor="tab:red")
    # 强制内存轴底部从 0 开始
    ax2.set_ylim(bottom=0)

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    # 美化
    ax1.grid(True, linestyle='--', alpha=0.4)
    plt.title("GPU Utilization and Memory Growth Over Time")
    plt.tight_layout()
    plt.savefig("/sgl-workspace/sglang-multi-model/gpu_util_memory_growth.png")
    plt.show()

if __name__ == "__main__":
    plot_gpu_and_memory_usage("/sgl-workspace/sglang-multi-model/gpu_usage_log.csv")