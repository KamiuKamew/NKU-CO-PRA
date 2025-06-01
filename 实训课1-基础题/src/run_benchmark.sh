 #!/bin/bash

# 清理之前的性能结果文件
rm -f performance_results.txt

echo "开始性能测试..."

# 运行CPU各版本
echo "运行CPU基准版本..."
./outputfile baseline

echo "运行CPU OpenMP版本..."
./outputfile openmp

echo "运行CPU分块优化版本..."
./outputfile block

echo "运行CPU MPI版本（4进程）..."
mpirun -np 4 ./outputfile mpi

# DCU性能分析
echo "运行DCU版本性能分析..."

echo "1. 使用rocm-smi监控DCU状态..."
rocm-smi > dcu_status.txt
rocm-smi --showuse > dcu_usage.txt
rocm-smi --showmemuse > dcu_memory.txt
rocm-smi --showtemp > dcu_temperature.txt

echo "2. 使用hipprof进行性能分析..."
hipprof ./outputfile_dcu > hipprof_basic.txt
hipprof --analysis-metrics all ./outputfile_dcu > hipprof_detailed.txt

echo "3. 运行DCU版本..."
./outputfile_dcu

# 生成性能对比图表
echo "生成性能对比图表..."
python plot_performance.py

echo "性能测试完成。结果文件："
echo "- performance_results.txt：原始性能数据"
echo "- performance_comparison_bar.png：性能柱状图"
echo "- performance_comparison_line.png：性能折线图"
echo "- speedup_comparison.png：加速比图"
echo "- dcu_status.txt：DCU状态信息"
echo "- dcu_usage.txt：DCU使用率信息"
echo "- dcu_memory.txt：DCU内存使用信息"
echo "- dcu_temperature.txt：DCU温度信息"
echo "- hipprof_basic.txt：基本性能分析结果"
echo "- hipprof_detailed.txt：详细性能分析结果"