#!/bin/bash

echo "=================================================="
echo "  基于DCU的MLP低轨卫星带宽预测性能评测系统"
echo "=================================================="

# 设置运行参数
EXEC_NAME="./mlp_dcu"
LOG_FILE="performance_log.txt"
RESULT_FILE="results.txt"

# 检查可执行文件
if [ ! -f "$EXEC_NAME" ]; then
    echo "错误: 可执行文件 $EXEC_NAME 不存在! 请先运行编译脚本。"
    exit 1
fi

# 检查数据文件
if [ ! -f "data/starlink_bw.json" ]; then
    echo "错误: 数据文件不存在!"
    exit 1
fi

# 清理之前的日志
> $LOG_FILE
> $RESULT_FILE

echo "开始性能测试..."
echo "日志文件: $LOG_FILE"
echo "结果文件: $RESULT_FILE"

# 记录系统信息
echo "=== 系统信息 ===" >> $LOG_FILE
date >> $LOG_FILE
echo "DCU 设备信息:" >> $LOG_FILE
rocm-smi 2>/dev/null >> $LOG_FILE || echo "rocm-smi 不可用" >> $LOG_FILE
echo "" >> $LOG_FILE

# 运行性能测试
echo "=== 运行DCU加速的MLP训练和推理测试 ===" | tee -a $LOG_FILE

# 开始计时
start_time=$(date +%s.%N)

# 运行程序并同时输出到终端和日志文件
$EXEC_NAME 2>&1 | tee -a $LOG_FILE

# 检查运行结果
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    end_time=$(date +%s.%N)
    total_time=$(echo "$end_time - $start_time" | bc -l 2>/dev/null || awk "BEGIN {print $end_time - $start_time}")
    
    echo "" | tee -a $LOG_FILE
    echo "✅ 程序运行成功!" | tee -a $LOG_FILE
    echo "总运行时间: ${total_time} 秒" | tee -a $LOG_FILE
    
    # 提取性能指标
    echo "=== 性能指标摘要 ===" | tee -a $RESULT_FILE
    echo "运行时间: $(date)" >> $RESULT_FILE
    echo "总运行时间: ${total_time} 秒" >> $RESULT_FILE
    echo "" >> $RESULT_FILE
    
    # 从日志中提取关键性能数据
    echo "提取关键性能指标..."
    
    # 训练时间
    train_time=$(grep "训练时间:" $LOG_FILE | awk '{print $2}' | head -1)
    if [ ! -z "$train_time" ]; then
        echo "训练时间: $train_time" >> $RESULT_FILE
    fi
    
    # 推理时间
    infer_time=$(grep "推理时间:" $LOG_FILE | awk '{print $2}' | head -1)
    if [ ! -z "$infer_time" ]; then
        echo "推理时间: $infer_time" >> $RESULT_FILE
    fi
    
    # 推理吞吐量
    throughput=$(grep "推理吞吐量:" $LOG_FILE | awk '{print $2}' | head -1)
    if [ ! -z "$throughput" ]; then
        echo "推理吞吐量: $throughput 样本/秒" >> $RESULT_FILE
    fi
    
    # MSE误差
    mse=$(grep "归一化MSE:" $LOG_FILE | awk '{print $2}' | head -1)
    if [ ! -z "$mse" ]; then
        echo "归一化MSE: $mse" >> $RESULT_FILE
    fi
    
    echo "" >> $RESULT_FILE
    echo "=== 详细性能分析 ===" >> $RESULT_FILE
    
    # 计算性能指标
    if [ ! -z "$train_time" ] && [ ! -z "$infer_time" ]; then
        echo "训练阶段耗时: $train_time ms" >> $RESULT_FILE
        echo "推理阶段耗时: $infer_time ms" >> $RESULT_FILE
        
        # 计算训练推理比
        if command -v bc >/dev/null 2>&1; then
            ratio=$(echo "scale=2; $train_time / $infer_time" | bc)
            echo "训练/推理时间比: $ratio" >> $RESULT_FILE
        fi
    fi
    
    # 数据处理统计
    data_points=$(grep "加载了" $LOG_FILE | awk '{print $2}' | head -1)
    if [ ! -z "$data_points" ]; then
        echo "数据点数量: $data_points" >> $RESULT_FILE
    fi
    
    train_samples=$(grep "训练集大小:" $LOG_FILE | awk '{print $2}' | head -1)
    test_samples=$(grep "测试集大小:" $LOG_FILE | awk '{print $4}' | head -1)
    if [ ! -z "$train_samples" ] && [ ! -z "$test_samples" ]; then
        echo "训练样本: $train_samples" >> $RESULT_FILE
        echo "测试样本: $test_samples" >> $RESULT_FILE
    fi
    
    echo "" >> $RESULT_FILE
    echo "=== 硬件资源利用 ===" >> $RESULT_FILE
    
    # 记录DCU状态
    echo "DCU状态信息:" >> $RESULT_FILE
    rocm-smi --showtemp --showpower --showmemuse 2>/dev/null >> $RESULT_FILE || echo "DCU状态信息不可用" >> $RESULT_FILE
    
    echo "" >> $RESULT_FILE
    echo "=== 预测精度评估 ===" >> $RESULT_FILE
    
    # 提取预测结果
    echo "预测结果样例:" >> $RESULT_FILE
    grep -A 10 "预测结果对比" $LOG_FILE >> $RESULT_FILE
    
else
    echo "❌ 程序运行失败!" | tee -a $LOG_FILE
    echo "请检查错误信息并重新编译运行。" | tee -a $LOG_FILE
    exit 1
fi

echo "" | tee -a $LOG_FILE
echo "=================================================="
echo "  性能测试完成"
echo "=================================================="
echo "详细日志: $LOG_FILE"
echo "性能摘要: $RESULT_FILE"

# 显示摘要结果
if [ -f "$RESULT_FILE" ]; then
    echo ""
    echo "=== 性能测试摘要 ==="
    cat $RESULT_FILE
fi

echo ""
echo "测试完成时间: $(date)" 