import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.cm as cm

# 读取性能数据
df = pd.read_csv("performance_results.txt", header=None, names=["Method", "Time"])

# 清理重复数据，保留每种方法的最好结果
df_clean = df.groupby("Method")["Time"].min().reset_index()

# 设置样式
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 12
sns.set_palette("husl")

# 创建对数坐标柱状图
plt.figure(figsize=(12, 8))
ax = plt.bar(df_clean["Method"], df_clean["Time"])
plt.yscale("log")
plt.title("Matrix Multiplication Performance Comparison (Log Scale)", fontsize=16)
plt.xlabel("Implementation Method", fontsize=14)
plt.ylabel("Average Execution Time (ms) - Log Scale", fontsize=14)
plt.xticks(rotation=45)

# 添加数值标签
for i, v in enumerate(df_clean["Time"]):
    plt.text(i, v * 1.1, f"{v:.1f}ms", ha="center", va="bottom", fontweight="bold")

# 添加网格线
plt.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("performance_comparison_bar_log.png", dpi=300, bbox_inches="tight")
plt.close()

# 创建对数坐标折线图
plt.figure(figsize=(12, 8))
plt.plot(df_clean["Method"], df_clean["Time"], marker="o", linewidth=3, markersize=10)
plt.yscale("log")
plt.title("Matrix Multiplication Performance Trend (Log Scale)", fontsize=16)
plt.xlabel("Implementation Method", fontsize=14)
plt.ylabel("Average Execution Time (ms) - Log Scale", fontsize=14)
plt.xticks(rotation=45)

# 添加数值标签
for x, y in zip(range(len(df_clean)), df_clean["Time"]):
    plt.text(x, y * 1.2, f"{y:.1f}ms", ha="center", va="bottom", fontweight="bold")

# 添加网格线
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("performance_comparison_line_log.png", dpi=300, bbox_inches="tight")
plt.close()

# 计算加速比
baseline_time = df_clean[df_clean["Method"] == "Baseline"]["Time"].values[0]
df_clean["Speedup"] = baseline_time / df_clean["Time"]

# 创建加速比对数坐标柱状图
plt.figure(figsize=(12, 8))
colors = cm.get_cmap("viridis")(np.linspace(0, 1, len(df_clean)))
bars = plt.bar(df_clean["Method"], df_clean["Speedup"], color=colors)
plt.yscale("log")
plt.title("Speedup Relative to Baseline (Log Scale)", fontsize=16)
plt.xlabel("Implementation Method", fontsize=14)
plt.ylabel("Speedup (x) - Log Scale", fontsize=14)
plt.xticks(rotation=45)

# 添加数值标签
for i, v in enumerate(df_clean["Speedup"]):
    plt.text(i, v * 1.1, f"{v:.1f}x", ha="center", va="bottom", fontweight="bold")

# 添加基准线
plt.axhline(y=1, color="red", linestyle="--", alpha=0.7, label="Baseline (1x)")
plt.legend()

# 添加网格线
plt.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("speedup_comparison_log.png", dpi=300, bbox_inches="tight")
plt.close()

# 创建综合对比图
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# 线性坐标性能对比
ax1.bar(df_clean["Method"], df_clean["Time"], color=colors)
ax1.set_title("Performance Comparison (Linear Scale)")
ax1.set_ylabel("Time (ms)")
ax1.tick_params(axis="x", rotation=45)
for i, v in enumerate(df_clean["Time"]):
    ax1.text(i, v + max(df_clean["Time"]) * 0.02, f"{v:.1f}", ha="center", va="bottom")

# 对数坐标性能对比
ax2.bar(df_clean["Method"], df_clean["Time"], color=colors)
ax2.set_yscale("log")
ax2.set_title("Performance Comparison (Log Scale)")
ax2.set_ylabel("Time (ms) - Log Scale")
ax2.tick_params(axis="x", rotation=45)
ax2.grid(True, alpha=0.3, axis="y")

# 线性坐标加速比
ax3.bar(df_clean["Method"], df_clean["Speedup"], color=colors)
ax3.set_title("Speedup (Linear Scale)")
ax3.set_ylabel("Speedup (x)")
ax3.tick_params(axis="x", rotation=45)
ax3.axhline(y=1, color="red", linestyle="--", alpha=0.7)

# 对数坐标加速比
ax4.bar(df_clean["Method"], df_clean["Speedup"], color=colors)
ax4.set_yscale("log")
ax4.set_title("Speedup (Log Scale)")
ax4.set_ylabel("Speedup (x) - Log Scale")
ax4.tick_params(axis="x", rotation=45)
ax4.axhline(y=1, color="red", linestyle="--", alpha=0.7)
ax4.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("comprehensive_performance_analysis.png", dpi=300, bbox_inches="tight")
plt.close()

# 打印性能统计信息
print("\nPerformance Statistics (Cleaned Data):")
print("=" * 70)
print(f"{'Method':<15} {'Time (ms)':<15} {'Speedup':<12} {'Efficiency':<12}")
print("-" * 70)
for _, row in df_clean.iterrows():
    efficiency = (
        (row["Speedup"] / 4) * 100
        if row["Method"] in ["OpenMP", "Block"]
        else row["Speedup"]
    )
    print(
        f"{row['Method']:<15} {row['Time']:<15.2f} {row['Speedup']:<12.2f}x {efficiency:<12.1f}%"
    )

print(
    f"\n最佳性能: {df_clean.loc[df_clean['Time'].idxmin(), 'Method']} ({df_clean['Time'].min():.2f} ms)"
)
print(
    f"最大加速比: {df_clean.loc[df_clean['Speedup'].idxmax(), 'Method']} ({df_clean['Speedup'].max():.2f}x)"
)
