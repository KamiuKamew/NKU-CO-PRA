#!/bin/bash

echo "Debug测试开始..."

# 清理之前的结果
rm -f performance_results.txt

echo "===== 测试基准版本 ====="
./outputfile baseline

echo ""
echo "===== 测试OpenMP版本 ====="
./outputfile openmp

echo ""
echo "===== 测试分块版本 ====="
./outputfile block

echo ""
echo "===== 测试MPI版本 ====="
mpirun --allow-run-as-root -np 4 ./outputfile mpi

echo ""
echo "===== 测试DCU版本 ====="
./outputfile_dcu

echo ""
echo "===== 检查性能结果文件 ====="
if [ -f "performance_results.txt" ]; then
    echo "性能结果文件内容："
    cat performance_results.txt
else
    echo "性能结果文件不存在"
fi

echo ""
echo "Debug测试完成" 