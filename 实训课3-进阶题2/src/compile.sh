#!/bin/bash

echo "编译进阶题2..."

g++ -O3 -std=c++17 -o predict_cpu compare_cpu.cpp
if [ $? -eq 0 ]; then
    echo "CPU版本编译成功"
else
    echo "CPU版本编译失败"
    exit 1
fi

g++ -O3 -std=c++17 -o predict_cpu_full compare_cpu_full.cpp
if [ $? -eq 0 ]; then
    echo "CPU完整版本编译成功"
else
    echo "CPU完整版本编译失败"
    exit 1
fi

if command -v hipcc &> /dev/null; then
    hipcc -O3 -std=c++17 -o predict_dcu mlp_dcu.cpp
    if [ $? -eq 0 ]; then
        echo "DCU版本编译成功"
    else
        echo "DCU版本编译失败"
    fi
else
    echo "HIP编译器未找到，跳过DCU编译"
fi

echo "编译完成" 