#!/bin/bash

echo "开始MLP性能测试..."

# 清理之前的结果
rm -f mlp_performance_results.txt

# 运行基本测试
echo "===== 基本功能测试 ====="
./mlp_forward

echo ""
echo "===== 性能分析测试 ====="

# 使用rocm-smi监控DCU状态
echo "1. DCU状态监控..."
rocm-smi > mlp_dcu_status.txt
rocm-smi --showuse > mlp_dcu_usage.txt
rocm-smi --showmemuse > mlp_dcu_memory.txt
rocm-smi --showtemp > mlp_dcu_temperature.txt

# 使用hipprof进行性能分析
echo "2. 性能剖析..."
hipprof ./mlp_forward > mlp_hipprof_basic.txt
hipprof --analysis-metrics all ./mlp_forward > mlp_hipprof_detailed.txt

echo ""
echo "===== 多次运行性能统计 ====="
echo "运行5次MLP测试获取平均性能..."

total_cpu_time=0
total_gpu_time=0
runs=5

for i in $(seq 1 $runs); do
    echo "第 $i 次运行..."
    ./mlp_forward > temp_output.txt
    
    # 提取时间数据
    cpu_time=$(grep "CPU Time:" temp_output.txt | awk '{print $3}')
    gpu_time=$(grep "DCU Time:" temp_output.txt | awk '{print $3}')
    
    total_cpu_time=$(echo "$total_cpu_time + $cpu_time" | bc -l)
    total_gpu_time=$(echo "$total_gpu_time + $gpu_time" | bc -l)
    
    echo "CPU: ${cpu_time}ms, DCU: ${gpu_time}ms"
done

# 计算平均值
avg_cpu_time=$(echo "scale=2; $total_cpu_time / $runs" | bc -l)
avg_gpu_time=$(echo "scale=2; $total_gpu_time / $runs" | bc -l)
avg_speedup=$(echo "scale=2; $avg_cpu_time / $avg_gpu_time" | bc -l)

echo ""
echo "===== 平均性能结果 ====="
echo "平均CPU时间: ${avg_cpu_time}ms"
echo "平均DCU时间: ${avg_gpu_time}ms"
echo "平均加速比: ${avg_speedup}x"

# 保存结果到文件
echo "CPU,$avg_cpu_time" > mlp_performance_results.txt
echo "DCU,$avg_gpu_time" >> mlp_performance_results.txt

# 清理临时文件
rm -f temp_output.txt

echo ""
echo "MLP性能测试完成。结果文件："
echo "- mlp_performance_results.txt：性能数据"
echo "- mlp_dcu_status.txt：DCU状态信息"
echo "- mlp_dcu_usage.txt：DCU使用率信息"
echo "- mlp_dcu_memory.txt：DCU内存使用信息"
echo "- mlp_dcu_temperature.txt：DCU温度信息"
echo "- mlp_hipprof_basic.txt：基本性能分析"
echo "- mlp_hipprof_detailed.txt：详细性能分析" 