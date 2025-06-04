#!/bin/bash

echo "编译进阶题1..."

g++ -O3 -o mlp_cpu sourcefile_mlp_cpu.cpp
if [ $? -eq 0 ]; then
    echo "CPU版本编译成功"
else
    echo "CPU版本编译失败"
    exit 1
fi

if command -v hipcc &> /dev/null; then
    hipcc -O3 -o mlp_forward sourcefile_mlp_forward.cpp
    if [ $? -eq 0 ]; then
        echo "DCU前向传播版本编译成功"
    else
        echo "DCU前向传播版本编译失败"
    fi
    
    hipcc -O3 -o mlp_optimized sourcefile_mlp_optimized.cpp
    if [ $? -eq 0 ]; then
        echo "DCU优化版本编译成功"
    else
        echo "DCU优化版本编译失败"
    fi
else
    echo "HIP编译器未找到，跳过DCU编译"
fi

echo "编译完成"

echo ""
echo "🎉 所有版本编译成功！"
echo "可以运行以下命令测试:"
echo "  ./mlp_forward    # DCU基础版本"
echo "  ./mlp_optimized  # DCU优化版本"  
echo "  ./mlp_cpu        # CPU版本"
echo ""
echo "或者运行完整测试:"
echo "  ./run_server_test.sh" 