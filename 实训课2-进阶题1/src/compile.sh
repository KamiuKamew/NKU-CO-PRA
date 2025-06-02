#!/bin/bash

echo "开始编译MLP前向传播..."

# 编译DCU版本
echo "编译DCU版本..."
hipcc sourcefile_mlp_forward.cpp -o mlp_forward
if [ $? -eq 0 ]; then
    echo "DCU版本编译成功"
else
    echo "DCU版本编译失败"
    exit 1
fi

echo "MLP编译完成" 