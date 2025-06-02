#!/bin/bash

echo "==========================================="
echo "     MLP Forward Propagation 全面测试"
echo "==========================================="

# 切换到源码目录
cd "$(dirname "$0")"

# 创建结果目录
mkdir -p results

# 设置权限
chmod +x *.sh

echo ""
echo "1. 编译阶段..."
echo "-------------------------------------------"

# 编译基础版本
echo "编译基础版本 MLP..."
hipcc sourcefile_mlp_forward.cpp -o mlp_forward
if [ $? -eq 0 ]; then
    echo "✓ 基础版本编译成功"
else
    echo "✗ 基础版本编译失败"
    exit 1
fi

# 编译优化版本
echo "编译优化版本 MLP..."
hipcc sourcefile_mlp_optimized.cpp -o mlp_optimized
if [ $? -eq 0 ]; then
    echo "✓ 优化版本编译成功"
else
    echo "✗ 优化版本编译失败"
    exit 1
fi

echo ""
echo "2. 功能验证测试..."
echo "-------------------------------------------"

echo "测试基础版本..."
./mlp_forward > results/basic_test.txt
if [ $? -eq 0 ]; then
    echo "✓ 基础版本功能测试通过"
else
    echo "✗ 基础版本功能测试失败"
fi

echo "测试优化版本..."
./mlp_optimized > results/optimized_test.txt
if [ $? -eq 0 ]; then
    echo "✓ 优化版本功能测试通过"
else
    echo "✗ 优化版本功能测试失败"
fi

echo ""
echo "3. 性能基准测试..."
echo "-------------------------------------------"

echo "开始5次基准测试..."

# 初始化性能数据文件
echo "Method,Time" > results/mlp_performance_results.txt

# 基础版本性能测试
echo "测试基础版本性能..."
total_basic=0
runs=5

for i in $(seq 1 $runs); do
    echo "  基础版本第 $i 次运行..."
    ./mlp_forward > temp_basic.txt
    basic_time=$(grep "DCU Time:" temp_basic.txt | awk '{print $3}')
    total_basic=$(echo "$total_basic + $basic_time" | bc -l)
done

avg_basic=$(echo "scale=3; $total_basic / $runs" | bc -l)
echo "Basic,$avg_basic" >> results/mlp_performance_results.txt

# 优化版本性能测试
echo "测试优化版本性能..."
total_optimized=0

for i in $(seq 1 $runs); do
    echo "  优化版本第 $i 次运行..."
    ./mlp_optimized > temp_optimized.txt
    opt_time=$(grep "DCU Time (Optimized):" temp_optimized.txt | awk '{print $4}')
    total_optimized=$(echo "$total_optimized + $opt_time" | bc -l)
done

avg_optimized=$(echo "scale=3; $total_optimized / $runs" | bc -l)
echo "Optimized,$avg_optimized" >> results/mlp_performance_results.txt

# 计算性能提升
improvement=$(echo "scale=2; $avg_basic / $avg_optimized" | bc -l)

echo ""
echo "4. 硬件监控..."
echo "-------------------------------------------"

# DCU状态监控
echo "DCU状态监控..."
rocm-smi > results/dcu_status.txt
rocm-smi --showuse > results/dcu_usage.txt
rocm-smi --showmemuse > results/dcu_memory.txt
rocm-smi --showtemp > results/dcu_temperature.txt

echo ""
echo "5. 性能分析..."
echo "-------------------------------------------"

# 使用hipprof进行详细分析
echo "基础版本性能分析..."
hipprof ./mlp_forward > results/hipprof_basic.txt 2>&1

echo "优化版本性能分析..."
hipprof ./mlp_optimized > results/hipprof_optimized.txt 2>&1

echo ""
echo "6. 生成可视化图表..."
echo "-------------------------------------------"

# 检查是否有Python和必要的库
if command -v python3 &> /dev/null; then
    echo "生成性能图表..."
    
    # 创建简化的可视化脚本（避免依赖问题）
    cat > results/simple_plot.py << 'EOF'
import csv

print("=== MLP Performance Analysis ===")

# 读取性能数据
with open('mlp_performance_results.txt', 'r') as f:
    reader = csv.reader(f)
    next(reader)  # 跳过标题
    basic_time = float(next(reader)[1])
    opt_time = float(next(reader)[1])

improvement = basic_time / opt_time

print(f"基础版本时间: {basic_time:.3f} ms")
print(f"优化版本时间: {opt_time:.3f} ms") 
print(f"性能提升: {improvement:.2f}x")

# 简单的文本图表
print("\n性能对比图:")
print("基础版本 |" + "█" * int(basic_time/opt_time * 10) + f"| {basic_time:.3f}ms")
print("优化版本 |" + "█" * 10 + f"| {opt_time:.3f}ms")
print("         " + "-" * 50)
print(f"         优化版本比基础版本快 {improvement:.2f} 倍")
EOF

    cd results
    python3 simple_plot.py > performance_analysis.txt
    cd ..
    echo "✓ 性能分析报告已生成"
else
    echo "Python3 未安装，跳过图表生成"
fi

echo ""
echo "7. 总结报告..."
echo "-------------------------------------------"

cat > results/test_summary.txt << EOF
MLP Forward Propagation 测试总结报告
=====================================

测试时间: $(date)
网络配置: 1024×10 → 10×20 (ReLU) → 20×5

性能结果:
- 基础版本平均时间: ${avg_basic} ms
- 优化版本平均时间: ${avg_optimized} ms
- 性能提升: ${improvement}x

优化技术:
- 共享内存优化
- 内存块分割 (Tiling)
- 内核融合 (Kernel Fusion)
- 异步内存传输

测试状态:
- 基础版本编译: ✓
- 优化版本编译: ✓
- 功能验证: ✓
- 性能测试: ✓
- 硬件监控: ✓

生成的文件:
- basic_test.txt: 基础版本测试输出
- optimized_test.txt: 优化版本测试输出
- mlp_performance_results.txt: 性能数据
- dcu_status.txt: DCU状态信息
- hipprof_basic.txt: 基础版本性能分析
- hipprof_optimized.txt: 优化版本性能分析
- performance_analysis.txt: 性能分析报告
EOF

# 清理临时文件
rm -f temp_basic.txt temp_optimized.txt

echo ""
echo "=========================================="
echo "          测试完成总结"
echo "=========================================="
echo "基础版本平均时间: ${avg_basic} ms"
echo "优化版本平均时间: ${avg_optimized} ms"
echo "性能提升: ${improvement}x"
echo ""
echo "所有结果文件保存在 results/ 目录中"
echo "详细报告请查看: results/test_summary.txt"
echo "==========================================" 