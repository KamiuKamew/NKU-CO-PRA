import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取性能数据
df = pd.read_csv("performance_results.txt", header=None, names=["Method", "Time"])

# 设置样式
plt.style.use("seaborn")
sns.set_palette("husl")

# 创建柱状图
plt.figure(figsize=(10, 6))
ax = sns.barplot(x="Method", y="Time", data=df)
plt.title("Matrix Multiplication Performance Comparison")
plt.xlabel("Implementation Method")
plt.ylabel("Average Execution Time (ms)")

# 添加数值标签
for i in ax.containers:
    ax.bar_label(i, fmt="%.2f")

# 保存柱状图
plt.savefig("performance_comparison_bar.png")
plt.close()

# 创建折线图
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="Method", y="Time", marker="o")
plt.title("Matrix Multiplication Performance Trend")
plt.xlabel("Implementation Method")
plt.ylabel("Average Execution Time (ms)")

# 添加数值标签
for x, y in zip(range(len(df)), df["Time"]):
    plt.text(x, y, f"{y:.2f}", ha="center", va="bottom")

# 保存折线图
plt.savefig("performance_comparison_line.png")
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
for i in ax.containers:
    ax.bar_label(i, fmt="%.2fx")

# 保存加速比图
plt.savefig("speedup_comparison.png")
plt.close()

# 打印性能统计信息
print("\nPerformance Statistics:")
print("=" * 50)
print(f"{'Method':<15} {'Time (ms)':<15} {'Speedup':<10}")
print("-" * 50)
for _, row in df.iterrows():
    print(f"{row['Method']:<15} {row['Time']:<15.2f} {row['Speedup']:<10.2f}x")
