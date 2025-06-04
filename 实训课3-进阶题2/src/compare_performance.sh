#!/bin/bash

echo "=============================================================="
echo "  MLP低轨卫星带宽预测：CPU vs DCU 性能对比测试"
echo "=============================================================="

# 设置文件和目录
CPU_SOURCE="compare_cpu.cpp"
DCU_SOURCE="src/mlp_dcu.cpp"
CPU_EXEC="./mlp_cpu"
DCU_EXEC="./mlp_dcu"
COMPARE_LOG="performance_comparison.txt"

# 检查数据文件
if [ ! -f "data/starlink_bw.json" ]; then
    echo "错误: 数据文件不存在!"
    exit 1
fi

# 清理之前的结果
> $COMPARE_LOG

echo "开始性能对比测试..." | tee -a $COMPARE_LOG
echo "测试时间: $(date)" | tee -a $COMPARE_LOG
echo "" | tee -a $COMPARE_LOG

# =========================== 编译阶段 ===========================

echo "=== 第一阶段：编译程序 ===" | tee -a $COMPARE_LOG

# 编译CPU版本
echo "编译CPU基准版本..." | tee -a $COMPARE_LOG
if [ -f "$CPU_SOURCE" ]; then
    g++ -O3 -std=c++14 $CPU_SOURCE -o mlp_cpu
    if [ $? -eq 0 ]; then
        echo "✅ CPU版本编译成功" | tee -a $COMPARE_LOG
    else
        echo "❌ CPU版本编译失败" | tee -a $COMPARE_LOG
        exit 1
    fi
else
    echo "❌ CPU源文件不存在: $CPU_SOURCE" | tee -a $COMPARE_LOG
    exit 1
fi

# 编译DCU版本
echo "编译DCU加速版本..." | tee -a $COMPARE_LOG
if [ -f "$DCU_SOURCE" ]; then
    hipcc -O3 -std=c++14 -DHIP_PLATFORM_AMD $DCU_SOURCE -o mlp_dcu
    if [ $? -eq 0 ]; then
        echo "✅ DCU版本编译成功" | tee -a $COMPARE_LOG
    else
        echo "❌ DCU版本编译失败" | tee -a $COMPARE_LOG
        echo "注意: 如果没有DCU环境，将只运行CPU测试" | tee -a $COMPARE_LOG
        DCU_AVAILABLE=false
    fi
else
    echo "❌ DCU源文件不存在: $DCU_SOURCE" | tee -a $COMPARE_LOG
    DCU_AVAILABLE=false
fi

echo "" | tee -a $COMPARE_LOG

# =========================== 系统信息收集 ===========================

echo "=== 第二阶段：系统信息收集 ===" | tee -a $COMPARE_LOG

# CPU信息
echo "CPU信息:" | tee -a $COMPARE_LOG
lscpu | grep "Model name" | tee -a $COMPARE_LOG 2>/dev/null || echo "CPU信息不可用" | tee -a $COMPARE_LOG
echo "CPU核心数: $(nproc)" | tee -a $COMPARE_LOG

# 内存信息
echo "内存信息:" | tee -a $COMPARE_LOG
free -h | head -2 | tee -a $COMPARE_LOG

# DCU信息
if [ "$DCU_AVAILABLE" != "false" ]; then
    echo "DCU信息:" | tee -a $COMPARE_LOG
    rocm-smi 2>/dev/null | tee -a $COMPARE_LOG || echo "DCU信息不可用" | tee -a $COMPARE_LOG
fi

echo "" | tee -a $COMPARE_LOG

# =========================== 性能测试阶段 ===========================

echo "=== 第三阶段：性能测试 ===" | tee -a $COMPARE_LOG

# CPU性能测试
echo ">>> 运行CPU基准测试..." | tee -a $COMPARE_LOG
cpu_start_time=$(date +%s.%N)

if [ -f "$CPU_EXEC" ]; then
    echo "--- CPU测试开始 ---" >> $COMPARE_LOG
    $CPU_EXEC >> cpu_results.txt 2>&1
    cpu_exit_code=$?
    echo "--- CPU测试结束 ---" >> $COMPARE_LOG
    
    cpu_end_time=$(date +%s.%N)
    cpu_total_time=$(echo "$cpu_end_time - $cpu_start_time" | bc -l 2>/dev/null || awk "BEGIN {print $cpu_end_time - $cpu_start_time}")
    
    if [ $cpu_exit_code -eq 0 ]; then
        echo "✅ CPU测试完成，总耗时: ${cpu_total_time} 秒" | tee -a $COMPARE_LOG
        
        # 提取CPU性能指标
        cpu_train_time=$(grep "训练时间:" cpu_results.txt | awk '{print $2}' | head -1)
        cpu_infer_time=$(grep "推理时间:" cpu_results.txt | awk '{print $2}' | head -1)
        cpu_throughput=$(grep "推理吞吐量:" cpu_results.txt | awk '{print $2}' | head -1)
        cpu_mse=$(grep "归一化MSE:" cpu_results.txt | awk '{print $2}' | head -1)
        
    else
        echo "❌ CPU测试失败" | tee -a $COMPARE_LOG
        cpu_train_time="失败"
        cpu_infer_time="失败"
        cpu_throughput="失败"
        cpu_mse="失败"
    fi
else
    echo "❌ CPU可执行文件不存在" | tee -a $COMPARE_LOG
    cpu_train_time="未测试"
    cpu_infer_time="未测试"
    cpu_throughput="未测试"
    cpu_mse="未测试"
fi

echo "" | tee -a $COMPARE_LOG

# DCU性能测试
if [ "$DCU_AVAILABLE" != "false" ] && [ -f "$DCU_EXEC" ]; then
    echo ">>> 运行DCU加速测试..." | tee -a $COMPARE_LOG
    dcu_start_time=$(date +%s.%N)
    
    echo "--- DCU测试开始 ---" >> $COMPARE_LOG
    $DCU_EXEC >> dcu_results.txt 2>&1
    dcu_exit_code=$?
    echo "--- DCU测试结束 ---" >> $COMPARE_LOG
    
    dcu_end_time=$(date +%s.%N)
    dcu_total_time=$(echo "$dcu_end_time - $dcu_start_time" | bc -l 2>/dev/null || awk "BEGIN {print $dcu_end_time - $dcu_start_time}")
    
    if [ $dcu_exit_code -eq 0 ]; then
        echo "✅ DCU测试完成，总耗时: ${dcu_total_time} 秒" | tee -a $COMPARE_LOG
        
        # 提取DCU性能指标
        dcu_train_time=$(grep "训练时间:" dcu_results.txt | awk '{print $2}' | head -1)
        dcu_infer_time=$(grep "推理时间:" dcu_results.txt | awk '{print $2}' | head -1)
        dcu_throughput=$(grep "推理吞吐量:" dcu_results.txt | awk '{print $2}' | head -1)
        dcu_mse=$(grep "归一化MSE:" dcu_results.txt | awk '{print $2}' | head -1)
        
    else
        echo "❌ DCU测试失败" | tee -a $COMPARE_LOG
        dcu_train_time="失败"
        dcu_infer_time="失败"
        dcu_throughput="失败"
        dcu_mse="失败"
    fi
else
    echo "⚠️  跳过DCU测试（DCU不可用或编译失败）" | tee -a $COMPARE_LOG
    dcu_train_time="未测试"
    dcu_infer_time="未测试"
    dcu_throughput="未测试"
    dcu_mse="未测试"
    dcu_total_time="未测试"
fi

echo "" | tee -a $COMPARE_LOG

# =========================== 性能对比分析 ===========================

echo "=== 第四阶段：性能对比分析 ===" | tee -a $COMPARE_LOG

# 创建对比表格
echo "性能对比表:" | tee -a $COMPARE_LOG
echo "┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐" | tee -a $COMPARE_LOG
echo "│     指标        │   CPU基准版本   │   DCU加速版本   │    加速比       │" | tee -a $COMPARE_LOG
echo "├─────────────────┼─────────────────┼─────────────────┼─────────────────┤" | tee -a $COMPARE_LOG

# 训练时间对比
if [ "$cpu_train_time" != "失败" ] && [ "$cpu_train_time" != "未测试" ] && 
   [ "$dcu_train_time" != "失败" ] && [ "$dcu_train_time" != "未测试" ]; then
    train_speedup=$(echo "scale=2; $cpu_train_time / $dcu_train_time" | bc -l 2>/dev/null || echo "N/A")
else
    train_speedup="N/A"
fi

printf "│ %-15s │ %-15s │ %-15s │ %-15s │\n" "训练时间(ms)" "$cpu_train_time" "$dcu_train_time" "${train_speedup}x" | tee -a $COMPARE_LOG

# 推理时间对比
if [ "$cpu_infer_time" != "失败" ] && [ "$cpu_infer_time" != "未测试" ] && 
   [ "$dcu_infer_time" != "失败" ] && [ "$dcu_infer_time" != "未测试" ]; then
    infer_speedup=$(echo "scale=2; $cpu_infer_time / $dcu_infer_time" | bc -l 2>/dev/null || echo "N/A")
else
    infer_speedup="N/A"
fi

printf "│ %-15s │ %-15s │ %-15s │ %-15s │\n" "推理时间(ms)" "$cpu_infer_time" "$dcu_infer_time" "${infer_speedup}x" | tee -a $COMPARE_LOG

# 吞吐量对比
printf "│ %-15s │ %-15s │ %-15s │ %-15s │\n" "吞吐量(样本/秒)" "$cpu_throughput" "$dcu_throughput" "-" | tee -a $COMPARE_LOG

# 精度对比
printf "│ %-15s │ %-15s │ %-15s │ %-15s │\n" "归一化MSE" "$cpu_mse" "$dcu_mse" "-" | tee -a $COMPARE_LOG

echo "└─────────────────┴─────────────────┴─────────────────┴─────────────────┘" | tee -a $COMPARE_LOG

echo "" | tee -a $COMPARE_LOG

# =========================== 详细分析 ===========================

echo "=== 详细性能分析 ===" | tee -a $COMPARE_LOG

# 训练性能分析
if [ "$train_speedup" != "N/A" ]; then
    echo "训练性能分析:" | tee -a $COMPARE_LOG
    echo "- DCU相对于CPU的训练加速比: ${train_speedup}x" | tee -a $COMPARE_LOG
    
    if [ $(echo "$train_speedup > 1" | bc -l 2>/dev/null || echo 0) -eq 1 ]; then
        echo "- ✅ DCU训练性能优于CPU" | tee -a $COMPARE_LOG
    else
        echo "- ⚠️  CPU训练性能优于DCU（可能由于数据规模较小）" | tee -a $COMPARE_LOG
    fi
fi

# 推理性能分析
if [ "$infer_speedup" != "N/A" ]; then
    echo "推理性能分析:" | tee -a $COMPARE_LOG
    echo "- DCU相对于CPU的推理加速比: ${infer_speedup}x" | tee -a $COMPARE_LOG
    
    if [ $(echo "$infer_speedup > 1" | bc -l 2>/dev/null || echo 0) -eq 1 ]; then
        echo "- ✅ DCU推理性能优于CPU" | tee -a $COMPARE_LOG
    else
        echo "- ⚠️  CPU推理性能优于DCU（可能由于批次较小）" | tee -a $COMPARE_LOG
    fi
fi

# 精度分析
if [ "$cpu_mse" != "失败" ] && [ "$cpu_mse" != "未测试" ] && 
   [ "$dcu_mse" != "失败" ] && [ "$dcu_mse" != "未测试" ]; then
    echo "精度分析:" | tee -a $COMPARE_LOG
    echo "- CPU MSE: $cpu_mse" | tee -a $COMPARE_LOG
    echo "- DCU MSE: $dcu_mse" | tee -a $COMPARE_LOG
    
    # 检查精度差异
    diff=$(echo "scale=6; sqrt(($cpu_mse - $dcu_mse)^2)" | bc -l 2>/dev/null || echo "N/A")
    if [ "$diff" != "N/A" ]; then
        echo "- 精度差异: $diff" | tee -a $COMPARE_LOG
        if [ $(echo "$diff < 0.001" | bc -l 2>/dev/null || echo 0) -eq 1 ]; then
            echo "- ✅ 两个版本精度基本一致" | tee -a $COMPARE_LOG
        else
            echo "- ⚠️  两个版本存在精度差异" | tee -a $COMPARE_LOG
        fi
    fi
fi

echo "" | tee -a $COMPARE_LOG

# =========================== 结论和建议 ===========================

echo "=== 测试结论和建议 ===" | tee -a $COMPARE_LOG

if [ "$train_speedup" != "N/A" ] && [ "$infer_speedup" != "N/A" ]; then
    echo "性能总结:" | tee -a $COMPARE_LOG
    echo "- 训练阶段: DCU ${train_speedup}x 于 CPU" | tee -a $COMPARE_LOG
    echo "- 推理阶段: DCU ${infer_speedup}x 于 CPU" | tee -a $COMPARE_LOG
    
    # 给出使用建议
    if [ $(echo "$train_speedup > 1" | bc -l 2>/dev/null || echo 0) -eq 1 ] || 
       [ $(echo "$infer_speedup > 1" | bc -l 2>/dev/null || echo 0) -eq 1 ]; then
        echo "建议: 在当前配置下，DCU在某些方面表现更好，适合计算密集型任务" | tee -a $COMPARE_LOG
    else
        echo "建议: 当前网络规模较小，CPU表现更好。对于大规模数据和网络，DCU优势会更明显" | tee -a $COMPARE_LOG
    fi
else
    echo "⚠️  无法完成完整的性能对比分析" | tee -a $COMPARE_LOG
fi

echo "" | tee -a $COMPARE_LOG
echo "测试完成时间: $(date)" | tee -a $COMPARE_LOG

# =========================== 清理和输出 ===========================

echo "" | tee -a $COMPARE_LOG
echo "=============================================================="
echo "  性能对比测试完成"
echo "=============================================================="
echo "详细结果已保存到: $COMPARE_LOG"
echo "CPU测试日志: cpu_results.txt"
if [ "$DCU_AVAILABLE" != "false" ]; then
    echo "DCU测试日志: dcu_results.txt"
fi

echo ""
echo "=== 性能对比摘要 ==="
cat $COMPARE_LOG | grep -A 20 "性能对比表:" 