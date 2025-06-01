#!/bin/bash

echo "开始编译..."

# 编译CPU版本
echo "编译CPU版本..."
mpic++ -fopenmp -o outputfile sourcefile.cpp
if [ $? -eq 0 ]; then
    echo "CPU版本编译成功"
else
    echo "CPU版本编译失败"
    exit 1
fi

# 编译DCU版本
echo "编译DCU版本..."
hipcc sourcefile_dcu.cpp -o outputfile_dcu
if [ $? -eq 0 ]; then
    echo "DCU版本编译成功"
else
    echo "DCU版本编译失败"
    exit 1
fi

echo "所有编译完成" 