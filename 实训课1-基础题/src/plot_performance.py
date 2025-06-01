import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取性能数据
df = pd.read_csv("performance_results.txt", header=None, names=["Method", "Time"])

# 设置样式
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 12
sns.set_palette("husl")

# 创建柱状图
plt.figure(figsize=(10, 6))
ax = sns.barplot(x="Method", y="Time", data=df)
plt.title("Matrix Multiplication Performance Comparison")
plt.xlabel("Implementation Method")
plt.ylabel("Average Execution Time (ms)")

# 添加数值标签
for i, v in enumerate(df["Time"]):
    ax.text(i, v + max(df["Time"]) * 0.01, f"{v:.2f}", ha="center", va="bottom")

# 保存柱状图
plt.tight_layout()
plt.savefig("performance_comparison_bar.png", dpi=300, bbox_inches="tight")
plt.close()

# 创建折线图
plt.figure(figsize=(10, 6))
plt.plot(df["Method"], df["Time"], marker="o", linewidth=2, markersize=8)
plt.title("Matrix Multiplication Performance Trend")
plt.xlabel("Implementation Method")
plt.ylabel("Average Execution Time (ms)")
plt.xticks(rotation=45)

# 添加数值标签
for x, y in zip(range(len(df)), df["Time"]):
    plt.text(x, y + max(df["Time"]) * 0.02, f"{y:.2f}", ha="center", va="bottom")

# 保存折线图
plt.tight_layout()
plt.savefig("performance_comparison_line.png", dpi=300, bbox_inches="tight")
plt.close()

# 计算加速比
baseline_time = df[df["Method"] == "Baseline"]["Time"].values[0]
df["Speedup"] = baseline_time / df["Time"]

# 创建加速比柱状图
plt.figure(figsize=(10, 6))
ax = sns.barplot(x="Method", y="Speedup", data=df)
plt.title("Speedup Relative to Baseline")
plt.xlabel("Implementation Method")
plt.ylabel("Speedup (x)")

# 添加数值标签
for i, v in enumerate(df["Speedup"]):
    ax.text(i, v + max(df["Speedup"]) * 0.01, f"{v:.2f}x", ha="center", va="bottom")

# 保存加速比图
plt.tight_layout()
plt.savefig("speedup_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

# 打印性能统计信息
print("\nPerformance Statistics:")
print("=" * 50)
print(f"{'Method':<15} {'Time (ms)':<15} {'Speedup':<10}")
print("-" * 50)
for _, row in df.iterrows():
    print(f"{row['Method']:<15} {row['Time']:<15.2f} {row['Speedup']:<10.2f}x")
