#!/bin/bash

echo "=== DCU MLP 神经网络编译脚本 ==="

# 设置编译参数
HIPCC_FLAGS="-O3 -std=c++14 -DHIP_PLATFORM_AMD"
SOURCE_FILE="src/mlp_dcu.cpp"
EXEC_NAME="mlp_dcu"

# 检查源文件是否存在
if [ ! -f "$SOURCE_FILE" ]; then
    echo "错误: 源文件 $SOURCE_FILE 不存在!"
    exit 1
fi

# 检查数据文件是否存在
if [ ! -f "data/starlink_bw.json" ]; then
    echo "错误: 数据文件 data/starlink_bw.json 不存在!"
    exit 1
fi

echo "编译源文件: $SOURCE_FILE"
echo "输出文件: $EXEC_NAME"
echo "编译参数: $HIPCC_FLAGS"

# 编译
hipcc $HIPCC_FLAGS $SOURCE_FILE -o $EXEC_NAME

# 检查编译结果
if [ $? -eq 0 ]; then
    echo "✅ 编译成功!"
    echo "可执行文件: $EXEC_NAME"
    ls -la $EXEC_NAME
else
    echo "❌ 编译失败!"
    exit 1
fi

echo "=== 编译完成 ===" 