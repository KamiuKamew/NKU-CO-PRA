#!/bin/bash

echo "编译基础题..."

mpic++ -fopenmp -O3 -o matmul_cpu sourcefile.cpp
if [ $? -eq 0 ]; then
    echo "CPU版本编译成功"
else
    echo "CPU版本编译失败"
    exit 1
fi

if command -v hipcc &> /dev/null; then
    hipcc -O3 -o matmul_dcu sourcefile_dcu.cpp
    if [ $? -eq 0 ]; then
        echo "DCU版本编译成功"
    else
        echo "DCU版本编译失败"
        exit 1
    fi
else
    echo "HIP编译器未找到，跳过DCU编译"
fi

echo "编译完成" 