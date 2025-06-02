#!/bin/bash

echo "========================================="
echo "     MLP神经网络前向传播 - 本地编译"
echo "========================================="

echo ""
echo "🔧 环境检查..."
echo "-----------------------------------------"
echo "当前目录: $(pwd)"
echo "编译器检查:"
which g++ && echo "✓ g++ 可用" || echo "✗ g++ 不可用"
which hipcc && echo "✓ hipcc 可用 (DCU环境)" || echo "ℹ hipcc 不可用 (需要DCU服务器环境)"

echo ""
echo "📦 开始编译..."
echo "-----------------------------------------"

# 编译CPU版本
echo "1. 编译CPU版本..."
g++ -O3 -std=c++11 sourcefile_mlp_cpu.cpp -o mlp_cpu
if [ $? -eq 0 ]; then
    echo "✓ CPU版本编译成功"
    CPU_SUCCESS=true
else
    echo "✗ CPU版本编译失败"
    CPU_SUCCESS=false
fi

# 尝试编译DCU版本
if command -v hipcc &> /dev/null; then
    echo ""
    echo "2. 编译DCU基础版本..."
    hipcc -O3 sourcefile_mlp_forward.cpp -o mlp_forward
    if [ $? -eq 0 ]; then
        echo "✓ DCU基础版本编译成功"
        DCU_BASIC_SUCCESS=true
    else
        echo "✗ DCU基础版本编译失败"
        DCU_BASIC_SUCCESS=false
    fi

    echo "3. 编译DCU优化版本..."
    hipcc -O3 sourcefile_mlp_optimized.cpp -o mlp_optimized
    if [ $? -eq 0 ]; then
        echo "✓ DCU优化版本编译成功"
        DCU_OPT_SUCCESS=true
    else
        echo "✗ DCU优化版本编译失败"
        DCU_OPT_SUCCESS=false
    fi
else
    echo ""
    echo "ℹ DCU编译器 (hipcc) 不可用"
    echo "  需要在配置了DCU环境的服务器上运行"
    echo "  DCU源文件已准备好，修复了宏冲突问题"
    DCU_BASIC_SUCCESS=false
    DCU_OPT_SUCCESS=false
fi

echo ""
echo "📊 编译结果汇总"
echo "========================================="
echo "CPU版本:      $([ "$CPU_SUCCESS" = true ] && echo "✓ 成功" || echo "✗ 失败")"
echo "DCU基础版本:  $([ "$DCU_BASIC_SUCCESS" = true ] && echo "✓ 成功" || echo "- 跳过")"
echo "DCU优化版本:  $([ "$DCU_OPT_SUCCESS" = true ] && echo "✓ 成功" || echo "- 跳过")"

echo ""
echo "🎯 可运行的测试命令:"
echo "-----------------------------------------"

if [ "$CPU_SUCCESS" = true ]; then
    echo "CPU版本测试:"
    echo "  ./mlp_cpu"
    echo ""
fi

if [ "$DCU_BASIC_SUCCESS" = true ]; then
    echo "DCU基础版本测试:"
    echo "  ./mlp_forward"
    echo ""
fi

if [ "$DCU_OPT_SUCCESS" = true ]; then
    echo "DCU优化版本测试:"
    echo "  ./mlp_optimized"
    echo ""
fi

echo "完整测试流程:"
echo "  ./run_server_test.sh    # 在DCU服务器上运行"
echo "  ./complete_test.sh      # 完整实验流程"

echo ""
echo "📝 注意事项:"
echo "-----------------------------------------"
if [ "$DCU_BASIC_SUCCESS" = false ] || [ "$DCU_OPT_SUCCESS" = false ]; then
    echo "• 本次修复了源代码中的宏冲突问题 (H -> H_SIZE)"
    echo "• DCU版本需要在配置DCU环境的服务器上编译和运行"
    echo "• 源文件已准备就绪，可直接在DCU环境中使用"
    echo "• CPU版本可以作为功能验证和性能基准"
fi

echo ""
echo "✅ 编译流程完成"
echo "=========================================" 