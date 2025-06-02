#!/bin/bash

echo "========================================="
echo "     MLP性能测试问题快速修复"
echo "========================================="

echo ""
echo "🔧 重新编译修复版本..."
echo "-----------------------------------------"

# 重新编译优化版本
echo "重新编译DCU优化版本..."
hipcc -O3 --gpu-max-threads-per-block=256 sourcefile_mlp_optimized.cpp -o mlp_optimized
if [ $? -eq 0 ]; then
    echo "✓ DCU优化版本重新编译成功"
else
    echo "✗ DCU优化版本重新编译失败"
fi

# 重新编译基础版本
echo "重新编译DCU基础版本..."
hipcc -O3 sourcefile_mlp_forward.cpp -o mlp_forward
if [ $? -eq 0 ]; then
    echo "✓ DCU基础版本重新编译成功"
else
    echo "✗ DCU基础版本重新编译失败"
fi

echo ""
echo "🧪 快速功能测试..."
echo "-----------------------------------------"

echo "测试CPU版本..."
./mlp_cpu | head -5

echo ""
echo "测试DCU基础版本..."
./mlp_forward | head -5

echo ""
echo "测试DCU优化版本..."
./mlp_optimized | head -5

echo ""
echo "📊 单次性能测试..."
echo "-----------------------------------------"

echo "CPU性能:"
./mlp_cpu | grep "Basic CPU Time:"

echo "DCU基础版本性能:"
./mlp_forward | grep "DCU Time:"

echo "DCU优化版本性能:"
./mlp_optimized | grep "DCU Time"

echo ""
echo "✅ 快速修复测试完成"
echo "现在可以运行: ./run_server_test.sh"
echo "=========================================" 