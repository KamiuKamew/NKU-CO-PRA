#!/usr/bin/env python3
"""
MLP性能测试结果可视化脚本
支持在有限环境中生成性能图表
"""

import sys
import os
import csv


def read_performance_data(filename):
    """读取性能数据"""
    data = []
    try:
        with open(filename, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(
                    {
                        "method": row["Method"],
                        "time": float(row["Time_ms"]),
                        "speedup": float(row["Speedup"]),
                    }
                )
        return data
    except Exception as e:
        print(f"读取性能数据失败: {e}")
        return []


def create_text_chart(data):
    """创建文本格式的性能图表"""
    if not data:
        return "无性能数据"

    # 找到最大时间用于归一化
    max_time = max(item["time"] for item in data)

    chart = []
    chart.append("=" * 80)
    chart.append("MLP 神经网络前向传播性能对比图")
    chart.append("=" * 80)
    chart.append("")

    # 执行时间对比
    chart.append("执行时间对比 (ms):")
    chart.append("-" * 60)
    for item in data:
        bar_length = int((item["time"] / max_time) * 40)
        bar = "█" * bar_length
        chart.append(f"{item['method']:15} |{bar:<40}| {item['time']:>8.3f}ms")

    chart.append("")

    # 加速比对比
    chart.append("加速比对比 (相对CPU):")
    chart.append("-" * 60)
    max_speedup = max(item["speedup"] for item in data)
    for item in data:
        bar_length = int((item["speedup"] / max_speedup) * 40)
        bar = "█" * bar_length
        chart.append(f"{item['method']:15} |{bar:<40}| {item['speedup']:>8.2f}x")

    chart.append("")
    chart.append("=" * 80)

    return "\n".join(chart)


def generate_performance_report(data):
    """生成性能分析报告"""
    if not data:
        return "无性能数据可分析"

    report = []
    report.append("# MLP神经网络前向传播性能分析报告")
    report.append("")
    report.append("## 性能数据概览")
    report.append("")

    # 性能表格
    report.append("| 实现方法 | 执行时间(ms) | 加速比 | 性能等级 |")
    report.append("|----------|-------------|--------|----------|")

    for item in data:
        if item["speedup"] >= 50:
            level = "🚀 极优"
        elif item["speedup"] >= 10:
            level = "⚡ 优秀"
        elif item["speedup"] >= 2:
            level = "👍 良好"
        else:
            level = "📊 基准"

        report.append(
            f"| {item['method']} | {item['time']:.3f} | {item['speedup']:.2f}x | {level} |"
        )

    report.append("")

    # 分析总结
    cpu_time = next((item["time"] for item in data if "CPU" in item["method"]), 0)
    dcu_items = [item for item in data if "DCU" in item["method"]]

    if dcu_items:
        best_dcu = min(dcu_items, key=lambda x: x["time"])
        report.append("## 关键发现")
        report.append("")
        report.append(
            f"- **最佳DCU性能**: {best_dcu['method']} 获得 {best_dcu['speedup']:.2f}x 加速比"
        )
        report.append(
            f"- **性能提升**: DCU相比CPU减少了 {((cpu_time - best_dcu['time']) / cpu_time * 100):.1f}% 的执行时间"
        )

        if len(dcu_items) > 1:
            basic_dcu = next(
                (item for item in dcu_items if "Basic" in item["method"]), None
            )
            opt_dcu = next(
                (item for item in dcu_items if "Optimized" in item["method"]), None
            )
            if basic_dcu and opt_dcu:
                opt_improvement = basic_dcu["time"] / opt_dcu["time"]
                report.append(
                    f"- **优化效果**: 优化版本比基础版本快 {opt_improvement:.2f}x"
                )

    report.append("")
    report.append("## 网络架构信息")
    report.append("- **输入层**: 1024 × 10")
    report.append("- **隐藏层**: 10 × 20 (ReLU激活)")
    report.append("- **输出层**: 20 × 5")
    report.append("- **总计算量**: ~307K 次浮点运算")

    return "\n".join(report)


def try_matplotlib_plot(data):
    """尝试使用matplotlib生成图表"""
    try:
        import matplotlib

        matplotlib.use("Agg")  # 使用非交互式后端
        import matplotlib.pyplot as plt
        import numpy as np

        plt.rcParams["font.size"] = 12
        plt.rcParams["figure.figsize"] = (12, 8)

        methods = [item["method"] for item in data]
        times = [item["time"] for item in data]
        speedups = [item["speedup"] for item in data]

        # 创建子图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]

        # 1. 执行时间柱状图
        bars1 = ax1.bar(methods, times, color=colors[: len(methods)])
        ax1.set_title("MLP Forward Propagation - Execution Time", fontweight="bold")
        ax1.set_ylabel("Time (ms)")
        ax1.set_xlabel("Implementation")
        for i, v in enumerate(times):
            ax1.text(
                i,
                v + max(times) * 0.02,
                f"{v:.3f}ms",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # 2. 对数坐标时间对比
        ax2.bar(methods, times, color=colors[: len(methods)])
        ax2.set_yscale("log")
        ax2.set_title("Execution Time (Log Scale)", fontweight="bold")
        ax2.set_ylabel("Time (ms) - Log Scale")
        ax2.set_xlabel("Implementation")
        ax2.grid(True, alpha=0.3, axis="y")

        # 3. 加速比对比
        bars3 = ax3.bar(methods, speedups, color=colors[: len(methods)])
        ax3.set_title("Speedup Relative to CPU", fontweight="bold")
        ax3.set_ylabel("Speedup (x)")
        ax3.set_xlabel("Implementation")
        ax3.axhline(y=1, color="red", linestyle="--", alpha=0.7, label="Baseline")
        for i, v in enumerate(speedups):
            ax3.text(
                i,
                v + max(speedups) * 0.02,
                f"{v:.2f}x",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # 4. 网络架构图
        ax4.axis("off")
        ax4.set_title("MLP Network Architecture", fontweight="bold")

        # 简化的网络结构图
        layer_info = ["Input\n1024×10", "Hidden\n10×20\n(ReLU)", "Output\n20×5"]
        positions = [0, 1, 2]

        for i, (pos, info) in enumerate(zip(positions, layer_info)):
            ax4.text(
                pos,
                0.5,
                info,
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor=colors[i % len(colors)],
                    alpha=0.7,
                ),
                fontweight="bold",
            )
            if i < len(positions) - 1:
                ax4.annotate(
                    "",
                    xy=(pos + 0.3, 0.5),
                    xytext=(pos + 0.15, 0.5),
                    arrowprops=dict(arrowstyle="->", lw=2, color="gray"),
                )

        ax4.set_xlim(-0.5, 2.5)
        ax4.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(
            "results/mlp_performance_visualization.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        return True

    except ImportError:
        return False
    except Exception as e:
        print(f"matplotlib绘图失败: {e}")
        return False


def main():
    """主函数"""
    # 检查是否在results目录中
    if not os.path.exists("performance_summary.txt"):
        if os.path.exists("results/performance_summary.txt"):
            os.chdir("results")
        else:
            print("错误: 找不到性能数据文件 performance_summary.txt")
            sys.exit(1)

    # 读取性能数据
    print("读取性能数据...")
    data = read_performance_data("performance_summary.txt")

    if not data:
        print("错误: 无法读取性能数据")
        sys.exit(1)

    print(f"成功读取 {len(data)} 个性能测试结果")

    # 生成文本图表
    print("\n生成文本格式性能图表...")
    text_chart = create_text_chart(data)

    # 保存文本图表
    with open("performance_text_chart.txt", "w", encoding="utf-8") as f:
        f.write(text_chart)

    print("✓ 文本图表已保存: performance_text_chart.txt")

    # 生成分析报告
    print("生成性能分析报告...")
    report = generate_performance_report(data)

    with open("performance_analysis_report.md", "w", encoding="utf-8") as f:
        f.write(report)

    print("✓ 分析报告已保存: performance_analysis_report.md")

    # 尝试生成matplotlib图表
    print("尝试生成图形化图表...")
    if try_matplotlib_plot(data):
        print("✓ 图形化图表已保存: mlp_performance_visualization.png")
    else:
        print("⚠ matplotlib不可用，跳过图形化图表生成")

    # 输出文本图表到控制台
    print("\n" + text_chart)

    print("\n🎉 可视化文件生成完成!")
    print("生成的文件:")
    print("- performance_text_chart.txt: 文本格式性能图表")
    print("- performance_analysis_report.md: 详细性能分析报告")
    if os.path.exists("mlp_performance_visualization.png"):
        print("- mlp_performance_visualization.png: 图形化性能图表")


if __name__ == "__main__":
    main()
