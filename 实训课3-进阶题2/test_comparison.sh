#!/bin/bash

echo "========================================================"
echo "  MLP 性能快速对比测试"
echo "========================================================"

# 编译两个版本
echo "编译CPU版本..."
g++ -O3 -std=c++14 compare_cpu.cpp -o mlp_cpu
if [ $? -ne 0 ]; then
    echo "❌ CPU版本编译失败"
    exit 1
fi

echo "编译DCU版本..."
bash compile_dcu.sh
if [ $? -ne 0 ]; then
    echo "❌ DCU版本编译失败"
    exit 1
fi

echo ""
echo "========================================================"
echo "  开始性能对比测试"
echo "========================================================"

# 运行CPU版本
echo "--- CPU版本测试 ---"
cpu_start=$(date +%s.%N)
./mlp_cpu > cpu_quick_test.log 2>&1
cpu_end=$(date +%s.%N)
cpu_time=$(echo "$cpu_end - $cpu_start" | bc -l 2>/dev/null || awk "BEGIN {print $cpu_end - $cpu_start}")

# 提取CPU关键指标
cpu_final_loss=$(grep "最终训练损失:" cpu_quick_test.log | awk '{print $2}' | head -1)
cpu_avg_error=$(grep "误差:" cpu_quick_test.log | awk '{print $6}' | sed 's/Mbps//' | awk '{sum+=$1; count++} END {print sum/count}')

echo "CPU总耗时: ${cpu_time} 秒"
echo "CPU最终损失: ${cpu_final_loss}"
echo "CPU平均误差: ${cpu_avg_error} Mbps"

echo ""

# 运行DCU版本
echo "--- DCU版本测试 ---"
dcu_start=$(date +%s.%N)
./mlp_dcu > dcu_quick_test.log 2>&1
dcu_end=$(date +%s.%N)
dcu_time=$(echo "$dcu_end - $dcu_start" | bc -l 2>/dev/null || awk "BEGIN {print $dcu_end - $dcu_start}")

# 提取DCU关键指标
dcu_final_loss=$(grep "最终训练损失:" dcu_quick_test.log | awk '{print $2}' | head -1)
dcu_avg_error=$(grep "误差:" dcu_quick_test.log | awk '{print $6}' | sed 's/Mbps//' | awk '{sum+=$1; count++} END {print sum/count}')

echo "DCU总耗时: ${dcu_time} 秒"
echo "DCU最终损失: ${dcu_final_loss}"
echo "DCU平均误差: ${dcu_avg_error} Mbps"

echo ""
echo "========================================================"
echo "  对比分析"
echo "========================================================"

# 计算性能比较
if [ ! -z "$cpu_time" ] && [ ! -z "$dcu_time" ]; then
    speedup=$(echo "scale=2; $cpu_time / $dcu_time" | bc -l 2>/dev/null || echo "N/A")
    echo "时间加速比: ${speedup}x (DCU相对CPU)"
fi

if [ ! -z "$cpu_final_loss" ] && [ ! -z "$dcu_final_loss" ]; then
    loss_ratio=$(echo "scale=3; $dcu_final_loss / $cpu_final_loss" | bc -l 2>/dev/null || echo "N/A")
    echo "损失比值: ${loss_ratio} (DCU/CPU, 越接近1越好)"
fi

if [ ! -z "$cpu_avg_error" ] && [ ! -z "$dcu_avg_error" ]; then
    error_ratio=$(echo "scale=2; $dcu_avg_error / $cpu_avg_error" | bc -l 2>/dev/null || echo "N/A")
    echo "误差比值: ${error_ratio} (DCU/CPU, 越接近1越好)"
fi

echo ""
echo "详细日志: cpu_quick_test.log, dcu_quick_test.log"
echo "测试完成时间: $(date)" 