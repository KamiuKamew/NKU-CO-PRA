#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLP Performance Analysis Script with Real Benchmarking Data
基于实际基准测试数据的MLP性能分析脚本 - 使用完整10000轮训练数据
"""

import matplotlib.pyplot as plt
import numpy as np
import re
from matplotlib import rcParams

# 设置字体兼容性
rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans"]
rcParams["axes.unicode_minus"] = False


def parse_dcu_performance_log(log_file):
    """解析DCU性能日志"""
    epochs = []
    losses = []
    prediction_errors = []
    actual_values = []
    predicted_values = []

    with open(log_file, "r", encoding="utf-8") as f:
        content = f.read()

    # 解析训练损失
    loss_pattern = r"Epoch (\d+)/\d+, 平均损失: ([\d.]+)"
    loss_matches = re.findall(loss_pattern, content)

    for epoch, loss in loss_matches:
        epochs.append(int(epoch))
        losses.append(float(loss))

    # 解析预测结果
    prediction_pattern = (
        r"样本 \d+ - 预测: ([\d.]+) Mbps, 实际: ([\d.]+) Mbps, 误差: ([\d.]+) Mbps"
    )
    pred_matches = re.findall(prediction_pattern, content)

    for pred, actual, error in pred_matches:
        predicted_values.append(float(pred))
        actual_values.append(float(actual))
        prediction_errors.append(float(error))

    # 解析性能指标
    train_time_match = re.search(r"训练时间: ([\d.]+) ms", content)
    infer_time_match = re.search(r"推理时间: ([\d.]+) ms", content)
    throughput_match = re.search(r"推理吞吐量: ([\d.]+) 样本/秒", content)
    final_loss_match = re.search(r"最终训练损失: ([\d.]+)", content)

    performance_metrics = {
        "train_time": float(train_time_match.group(1)) if train_time_match else 0,
        "infer_time": float(infer_time_match.group(1)) if infer_time_match else 0,
        "throughput": float(throughput_match.group(1)) if throughput_match else 0,
        "final_loss": float(final_loss_match.group(1)) if final_loss_match else 0,
    }

    return {
        "epochs": epochs,
        "losses": losses,
        "predicted_values": predicted_values,
        "actual_values": actual_values,
        "prediction_errors": prediction_errors,
        "performance_metrics": performance_metrics,
    }


def parse_cpu_performance_log(log_file):
    """解析CPU性能日志 - 使用完整训练数据"""
    epochs = []
    losses = []
    prediction_errors = []
    actual_values = []
    predicted_values = []

    with open(log_file, "r", encoding="utf-8") as f:
        content = f.read()

    # 解析训练损失 (CPU版本的输出格式)
    loss_pattern = r"Epoch (\d+)/\d+, 平均损失: ([\d.]+)"
    loss_matches = re.findall(loss_pattern, content)

    for epoch, loss in loss_matches:
        epochs.append(int(epoch))
        losses.append(float(loss))

    # 解析预测结果
    prediction_pattern = (
        r"样本 \d+ - 预测: ([\d.]+) Mbps, 实际: ([\d.]+) Mbps, 误差: (\d+) Mbps"
    )
    pred_matches = re.findall(prediction_pattern, content)

    for pred, actual, error in pred_matches:
        predicted_values.append(float(pred))
        actual_values.append(float(actual))
        prediction_errors.append(float(error))

    # 解析性能指标
    train_time_match = re.search(r"训练时间: ([\d.]+) ms", content)
    infer_time_match = re.search(r"推理时间: ([\d.]+) ms", content)
    throughput_match = re.search(r"推理吞吐量: ([\d.]+) 样本/秒", content)
    final_loss_match = re.search(r"最终训练损失: ([\d.]+)", content)

    performance_metrics = {
        "train_time": float(train_time_match.group(1)) if train_time_match else 0,
        "infer_time": float(infer_time_match.group(1)) if infer_time_match else 0,
        "throughput": float(throughput_match.group(1)) if throughput_match else 0,
        "final_loss": float(final_loss_match.group(1)) if final_loss_match else 0,
    }

    return {
        "epochs": epochs,
        "losses": losses,
        "predicted_values": predicted_values,
        "actual_values": actual_values,
        "prediction_errors": prediction_errors,
        "performance_metrics": performance_metrics,
    }


def compute_validation_accuracy(predicted, actual, tolerance_percent=20):
    """计算验证精度 - 在允许误差范围内的预测准确率"""
    correct_predictions = 0
    total_predictions = len(predicted)

    for i in range(total_predictions):
        error_percent = abs(predicted[i] - actual[i]) / actual[i] * 100
        if error_percent <= tolerance_percent:
            correct_predictions += 1

    return (
        (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    )


def create_training_process_visualizations(dcu_data, cpu_data):
    """创建与训练过程密切相关的可视化图表"""
    fig = plt.figure(figsize=(16, 12))

    # 1. 训练损失收敛对比 (主要图表) - 完整10000轮
    plt.subplot(2, 3, 1)
    if len(dcu_data["epochs"]) > 0:
        plt.plot(
            dcu_data["epochs"],
            dcu_data["losses"],
            "b-",
            linewidth=2,
            alpha=0.8,
            label="DCU Training Loss",
        )
    if len(cpu_data["epochs"]) > 0:
        plt.plot(
            cpu_data["epochs"],
            cpu_data["losses"],
            "r-",
            linewidth=2,
            alpha=0.8,
            label="CPU Training Loss (Full 10000 epochs)",
        )

    plt.xlabel("Training Epochs")
    plt.ylabel("Training Loss (MSE)")
    plt.title("Training Loss Convergence Comparison (10000 Epochs)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale("log")

    # 2. 验证精度对比
    plt.subplot(2, 3, 2)
    accuracy_tolerances = [5, 10, 15, 20, 25, 30]
    dcu_accuracies = []
    cpu_accuracies = []

    for tolerance in accuracy_tolerances:
        if len(dcu_data["predicted_values"]) > 0:
            dcu_acc = compute_validation_accuracy(
                dcu_data["predicted_values"], dcu_data["actual_values"], tolerance
            )
            dcu_accuracies.append(dcu_acc)
        else:
            dcu_accuracies.append(0)

        if len(cpu_data["predicted_values"]) > 0:
            cpu_acc = compute_validation_accuracy(
                cpu_data["predicted_values"], cpu_data["actual_values"], tolerance
            )
            cpu_accuracies.append(cpu_acc)
        else:
            cpu_accuracies.append(0)

    plt.plot(
        accuracy_tolerances,
        dcu_accuracies,
        "bo-",
        linewidth=2,
        markersize=6,
        label="DCU Accuracy",
    )
    plt.plot(
        accuracy_tolerances,
        cpu_accuracies,
        "ro-",
        linewidth=2,
        markersize=6,
        label="CPU Accuracy",
    )
    plt.xlabel("Error Tolerance (%)")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("Validation Accuracy vs Error Tolerance")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3. 预测误差分布对比
    plt.subplot(2, 3, 3)
    if len(dcu_data["prediction_errors"]) > 0:
        sample_indices = range(1, len(dcu_data["prediction_errors"]) + 1)
        plt.bar(
            [i - 0.2 for i in sample_indices],
            dcu_data["prediction_errors"],
            width=0.4,
            color="skyblue",
            alpha=0.7,
            label="DCU Errors",
        )

    if len(cpu_data["prediction_errors"]) > 0:
        sample_indices = range(1, len(cpu_data["prediction_errors"]) + 1)
        plt.bar(
            [i + 0.2 for i in sample_indices],
            cpu_data["prediction_errors"],
            width=0.4,
            color="lightcoral",
            alpha=0.7,
            label="CPU Errors",
        )

    plt.xlabel("Test Sample Index")
    plt.ylabel("Prediction Error (Mbps)")
    plt.title("Prediction Error Distribution Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 4. 性能指标对比 (基于实际测试数据)
    plt.subplot(2, 3, 4)
    categories = ["Training Time (s)", "Inference Time (ms)", "Throughput (samples/s)"]

    # 实际测试数据
    dcu_values = [
        dcu_data["performance_metrics"]["train_time"] / 1000,  # 转换为秒
        dcu_data["performance_metrics"]["infer_time"],
        dcu_data["performance_metrics"]["throughput"],
    ]
    cpu_values = [
        cpu_data["performance_metrics"]["train_time"] / 1000,  # 转换为秒
        cpu_data["performance_metrics"]["infer_time"],
        cpu_data["performance_metrics"]["throughput"],
    ]

    x = np.arange(len(categories))
    width = 0.35

    plt.bar(
        x - width / 2, cpu_values, width, label="CPU Full Training", color="lightcoral"
    )
    plt.bar(x + width / 2, dcu_values, width, label="DCU Training", color="skyblue")

    plt.xlabel("Performance Metrics")
    plt.ylabel("Values")
    plt.title("DCU vs CPU Performance (Real Full Training)")
    plt.xticks(x, categories, rotation=15)
    plt.legend()
    plt.yscale("log")
    plt.grid(True, alpha=0.3)

    # 5. 最终收敛质量对比
    plt.subplot(2, 3, 5)
    metrics = ["Final Loss", "Avg Error (Mbps)", "Max Error (Mbps)"]

    dcu_final_metrics = [
        dcu_data["performance_metrics"]["final_loss"],
        (
            np.mean(dcu_data["prediction_errors"])
            if len(dcu_data["prediction_errors"]) > 0
            else 0
        ),
        (
            max(dcu_data["prediction_errors"])
            if len(dcu_data["prediction_errors"]) > 0
            else 0
        ),
    ]
    cpu_final_metrics = [
        cpu_data["performance_metrics"]["final_loss"],
        (
            np.mean(cpu_data["prediction_errors"])
            if len(cpu_data["prediction_errors"]) > 0
            else 0
        ),
        (
            max(cpu_data["prediction_errors"])
            if len(cpu_data["prediction_errors"]) > 0
            else 0
        ),
    ]

    x = np.arange(len(metrics))
    width = 0.35

    plt.bar(x - width / 2, cpu_final_metrics, width, label="CPU", color="lightcoral")
    plt.bar(x + width / 2, dcu_final_metrics, width, label="DCU", color="skyblue")

    plt.xlabel("Quality Metrics")
    plt.ylabel("Values")
    plt.title("Final Training Quality Comparison")
    plt.xticks(x, metrics, rotation=15)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 6. 训练效率和加速比分析
    plt.subplot(2, 3, 6)
    speedup_metrics = ["Training", "Inference", "Throughput"]
    speedups = [
        cpu_values[0] / dcu_values[0] if dcu_values[0] > 0 else 0,  # 训练加速比
        cpu_values[1] / dcu_values[1] if dcu_values[1] > 0 else 0,  # 推理加速比
        dcu_values[2] / cpu_values[2] if cpu_values[2] > 0 else 0,  # 吞吐量提升比
    ]

    colors = ["red" if s < 1 else "green" for s in speedups]
    bars = plt.bar(speedup_metrics, speedups, color=colors, alpha=0.7)
    plt.axhline(
        y=1, color="black", linestyle="--", alpha=0.5, label="Baseline (CPU=DCU)"
    )
    plt.xlabel("Performance Metrics")
    plt.ylabel("Speedup Ratio")
    plt.title("DCU Speedup vs CPU (Full 10000 Epochs)")
    plt.legend()

    # 添加数值标签
    for bar, speedup in zip(bars, speedups):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.05,
            f"{speedup:.2f}x",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(
        "report/performance_analysis_real_full.png", dpi=300, bbox_inches="tight"
    )
    return fig


def generate_performance_summary_real(dcu_data, cpu_data):
    """生成基于实际完整训练数据的性能总结报告"""

    # 计算加速比
    train_speedup = (
        cpu_data["performance_metrics"]["train_time"]
        / dcu_data["performance_metrics"]["train_time"]
        if dcu_data["performance_metrics"]["train_time"] > 0
        else 0
    )

    infer_speedup = (
        cpu_data["performance_metrics"]["infer_time"]
        / dcu_data["performance_metrics"]["infer_time"]
        if dcu_data["performance_metrics"]["infer_time"] > 0
        else 0
    )

    throughput_improvement = (
        dcu_data["performance_metrics"]["throughput"]
        / cpu_data["performance_metrics"]["throughput"]
        if cpu_data["performance_metrics"]["throughput"] > 0
        else 0
    )

    summary = f"""
# MLP Performance Comparison - Real Full Training Benchmark Results

## 实际完整训练对比 (Both 10000 Epochs)

### Training Performance Comparison
- **DCU Training Time**: {dcu_data["performance_metrics"]["train_time"]:.2f} ms ({dcu_data["performance_metrics"]["train_time"]/1000:.2f} seconds)
- **CPU Training Time**: {cpu_data["performance_metrics"]["train_time"]:.2f} ms ({cpu_data["performance_metrics"]["train_time"]/1000:.2f} seconds)
- **Training Speedup**: {train_speedup:.3f}x {'(DCU faster)' if train_speedup > 1 else '(CPU faster)'}

### Final Training Quality (After 10000 Epochs)
- **DCU Final Loss**: {dcu_data["performance_metrics"]["final_loss"]:.6f}
- **CPU Final Loss**: {cpu_data["performance_metrics"]["final_loss"]:.6f}
- **DCU Training Epochs**: {max(dcu_data["epochs"]) if len(dcu_data["epochs"]) > 0 else 0}
- **CPU Training Epochs**: {max(cpu_data["epochs"]) if len(cpu_data["epochs"]) > 0 else 0}
- **Quality Winner**: {'DCU' if dcu_data["performance_metrics"]["final_loss"] < cpu_data["performance_metrics"]["final_loss"] else 'CPU'} (lower loss is better)

### Inference Performance Comparison
- **DCU Inference Time**: {dcu_data["performance_metrics"]["infer_time"]:.3f} ms (10 samples)
- **CPU Inference Time**: {cpu_data["performance_metrics"]["infer_time"]:.3f} ms (10 samples)
- **DCU Throughput**: {dcu_data["performance_metrics"]["throughput"]:.2f} samples/sec
- **CPU Throughput**: {cpu_data["performance_metrics"]["throughput"]:.2f} samples/sec
- **Inference Speedup**: {infer_speedup:.2f}x {'(DCU faster)' if infer_speedup > 1 else '(CPU faster)'}
- **Throughput Improvement**: {throughput_improvement:.2f}x

### Prediction Accuracy Comparison
#### DCU Results:
- **Average Error**: {np.mean(dcu_data["prediction_errors"]):.2f} Mbps
- **Min Error**: {min(dcu_data["prediction_errors"]):.2f} Mbps
- **Max Error**: {max(dcu_data["prediction_errors"]):.2f} Mbps
- **Error Std Dev**: {np.std(dcu_data["prediction_errors"]):.2f} Mbps

#### CPU Results:
- **Average Error**: {np.mean(cpu_data["prediction_errors"]):.2f} Mbps
- **Min Error**: {min(cpu_data["prediction_errors"]):.2f} Mbps
- **Max Error**: {max(cpu_data["prediction_errors"]):.2f} Mbps
- **Error Std Dev**: {np.std(cpu_data["prediction_errors"]):.2f} Mbps

### Key Findings (Fair 10000-Epoch Comparison)
1. **Training Performance**: {'DCU is ' + f'{train_speedup:.3f}x faster' if train_speedup > 1 else f'CPU is {1/train_speedup:.3f}x faster'} in training
2. **Inference Performance**: {'DCU achieves ' + f'{infer_speedup:.2f}x speedup' if infer_speedup > 1 else f'CPU is {1/infer_speedup:.2f}x faster'} in inference tasks
3. **Training Convergence**: {'DCU achieves better final loss (' + f'{dcu_data["performance_metrics"]["final_loss"]:.6f}' + ')' if dcu_data["performance_metrics"]["final_loss"] < cpu_data["performance_metrics"]["final_loss"] else 'CPU achieves better final loss (' + f'{cpu_data["performance_metrics"]["final_loss"]:.6f}' + ')'}
4. **Prediction Quality**: {'DCU' if np.mean(dcu_data["prediction_errors"]) < np.mean(cpu_data["prediction_errors"]) else 'CPU'} shows better average prediction accuracy

### Technical Analysis
- **DCU Advantages**: {('Superior training and inference performance' if train_speedup > 1 and infer_speedup > 1 else 'Superior inference performance' if infer_speedup > 1 else 'Training optimization needed')}
- **CPU Advantages**: {('Faster training and comparable inference' if train_speedup < 1 else 'Better optimization for this specific workload')}
- **Overall Assessment**: {('DCU shows clear performance advantages' if train_speedup > 1 else 'CPU implementation is currently more optimized for this workload')}

### Performance Summary
- **Training Speed**: CPU is {1/train_speedup:.2f}x faster
- **Inference Speed**: {'DCU' if infer_speedup > 1 else 'CPU'} is {max(infer_speedup, 1/infer_speedup):.2f}x faster
- **Final Loss Quality**: {'DCU' if dcu_data["performance_metrics"]["final_loss"] < cpu_data["performance_metrics"]["final_loss"] else 'CPU'} achieves better convergence
- **Prediction Accuracy**: {'DCU' if np.mean(dcu_data["prediction_errors"]) < np.mean(cpu_data["prediction_errors"]) else 'CPU'} has lower average error

### Optimization Recommendations
1. **For DCU**: Focus on training optimization - current training is {1/train_speedup:.2f}x slower than CPU
2. **For CPU**: {'Inference optimization could benefit from DCU techniques' if infer_speedup > 1 else 'Both platforms well-balanced for this workload'}
3. **Architecture**: Consider hybrid approach leveraging CPU for training and DCU for inference
"""

    return summary


def main():
    """主函数"""
    print("开始基于实际完整训练数据的MLP性能分析...")

    # 解析DCU性能日志
    try:
        dcu_data = parse_dcu_performance_log("performance_log.txt")
        print(
            f"成功解析DCU数据: {len(dcu_data['epochs'])} 个训练epoch, {len(dcu_data['predicted_values'])} 个预测结果"
        )
    except Exception as e:
        print(f"解析DCU日志失败: {e}")
        dcu_data = {
            "epochs": [],
            "losses": [],
            "predicted_values": [],
            "actual_values": [],
            "prediction_errors": [],
            "performance_metrics": {
                "train_time": 0,
                "infer_time": 0,
                "throughput": 0,
                "final_loss": 0,
            },
        }

    # 解析CPU完整性能日志
    try:
        cpu_data = parse_cpu_performance_log("cpu_full_performance_log.txt")
        print(
            f"成功解析CPU完整数据: {len(cpu_data['epochs'])} 个训练epoch, {len(cpu_data['predicted_values'])} 个预测结果"
        )
    except Exception as e:
        print(f"解析CPU完整日志失败: {e}")
        cpu_data = {
            "epochs": [],
            "losses": [],
            "predicted_values": [],
            "actual_values": [],
            "prediction_errors": [],
            "performance_metrics": {
                "train_time": 0,
                "infer_time": 0,
                "throughput": 0,
                "final_loss": 0,
            },
        }

    # 创建可视化图表
    print("生成完整训练数据性能对比可视化图表...")
    try:
        fig = create_training_process_visualizations(dcu_data, cpu_data)
        print("图表保存至: report/performance_analysis_real_full.png")
    except Exception as e:
        print(f"生成图表失败: {e}")
        return

    # 生成性能总结报告
    print("生成完整训练数据性能总结报告...")
    summary = generate_performance_summary_real(dcu_data, cpu_data)

    # 保存总结报告
    with open("report/performance_summary_real_full.md", "w", encoding="utf-8") as f:
        f.write(summary)

    print("性能总结报告保存至: report/performance_summary_real_full.md")
    print("基于实际完整训练数据的性能分析完成!")

    # 输出核心结果到控制台
    print("\n" + "=" * 60)
    print("核心性能对比结果 (公平的10000轮训练对比):")
    print("=" * 60)
    train_speedup = (
        cpu_data["performance_metrics"]["train_time"]
        / dcu_data["performance_metrics"]["train_time"]
    )
    infer_speedup = (
        cpu_data["performance_metrics"]["infer_time"]
        / dcu_data["performance_metrics"]["infer_time"]
    )
    print(
        f"训练加速比: {train_speedup:.3f}x {'(DCU更快)' if train_speedup > 1 else '(CPU更快)'}"
    )
    print(
        f"推理加速比: {infer_speedup:.2f}x {'(DCU更快)' if infer_speedup > 1 else '(CPU更快)'}"
    )
    print(f"DCU最终损失: {dcu_data['performance_metrics']['final_loss']:.6f}")
    print(f"CPU最终损失: {cpu_data['performance_metrics']['final_loss']:.6f}")
    print(f"DCU平均误差: {np.mean(dcu_data['prediction_errors']):.2f} Mbps")
    print(f"CPU平均误差: {np.mean(cpu_data['prediction_errors']):.2f} Mbps")
    print(
        f"收敛质量获胜者: {'DCU' if dcu_data['performance_metrics']['final_loss'] < cpu_data['performance_metrics']['final_loss'] else 'CPU'}"
    )


if __name__ == "__main__":
    main()
