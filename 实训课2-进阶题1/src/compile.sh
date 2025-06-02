#!/bin/bash

echo "开始编译MLP前向传播..."

echo "编译DCU基础版本..."
hipcc -O3 sourcefile_mlp_forward.cpp -o mlp_forward
if [ $? -eq 0 ]; then
    echo "✓ DCU基础版本编译成功"
else
    echo "✗ DCU基础版本编译失败"
    exit 1
fi

echo "编译DCU优化版本..."
hipcc -O3 sourcefile_mlp_optimized.cpp -o mlp_optimized
if [ $? -eq 0 ]; then
    echo "✓ DCU优化版本编译成功"
else
    echo "✗ DCU优化版本编译失败"
    exit 1
fi

echo "编译CPU版本..."
g++ -O3 -std=c++11 sourcefile_mlp_cpu.cpp -o mlp_cpu
if [ $? -eq 0 ]; then
    echo "✓ CPU版本编译成功"
else
    echo "✗ CPU版本编译失败"
    exit 1
fi

echo ""
echo "🎉 所有版本编译成功！"
echo "可以运行以下命令测试:"
echo "  ./mlp_forward    # DCU基础版本"
echo "  ./mlp_optimized  # DCU优化版本"  
echo "  ./mlp_cpu        # CPU版本"
echo ""
echo "或者运行完整测试:"
echo "  ./run_server_test.sh" 