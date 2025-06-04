# LEO 卫星网络带宽预测系统性能优化

南开大学计算机学院 智能计算创新设计赛（先导杯）实验项目

## 项目概述

本项目通过三个递进阶段实现 LEO 卫星网络带宽预测系统的性能优化：

- **基础题**：矩阵乘法优化（5 种实现方法）
- **进阶题 1**：基于矩阵乘法的多层感知机前向传播
- **进阶题 2**：完整的 LEO 卫星网络带宽预测系统

## 环境要求

### 必需

- Linux 操作系统（Ubuntu/CentOS）
- GCC 编译器（支持 C++17）
- MPI 编译器（mpic++）
- OpenMP 支持

### 可选（DCU 加速）

- HIP 编译器（hipcc）
- 曙光 DCU 或 AMD GPU

## 快速开始

### 方法 1：使用统一脚本

```bash
# 运行所有题目
./run.sh

# 或者单独运行
./run.sh basic      # 基础题
./run.sh advanced1  # 进阶题1
./run.sh advanced2  # 进阶题2
```

### 方法 2：手动编译运行

#### 基础题：矩阵乘法优化

```bash
cd 实训课1-基础题/src
./compile.sh

# 运行不同版本
./matmul_cpu baseline
./matmul_cpu openmp
./matmul_cpu block
mpirun -np 4 ./matmul_cpu mpi

# DCU版本（如果支持）
./matmul_dcu
```

#### 进阶题 1：MLP 前向传播

```bash
cd 实训课2-进阶题1/src
./compile.sh

# 运行测试
./mlp_cpu
./mlp_forward     # DCU版本
./mlp_optimized   # DCU优化版本
```

#### 进阶题 2：带宽预测

```bash
cd 实训课3-进阶题2/src
./compile.sh

# 运行预测系统
./predict_cpu
./predict_cpu_full
./predict_dcu     # DCU版本
```

## 性能结果

### 基础题性能对比

- **Baseline**: 18,373.2ms (1.00x)
- **OpenMP**: 1,607.0ms (11.43x 加速)
- **Block Tiling**: 1,505.2ms (12.21x 加速)
- **MPI**: 20,187.0ms (0.91x)
- **DCU**: 10.2ms (1,801.29x 加速)

### 进阶题 1 性能对比

- **CPU 基准**: 0.375ms
- **DCU 基础**: 0.664ms (0.56x)
- **DCU 优化**: 0.145ms (2.59x 加速)

### 进阶题 2 应用效果

- 训练数据：2,707 个样本
- 测试数据：677 个样本
- CPU 在小规模任务中表现优于 DCU

## 文件结构

```
NKU-CO-PRA/
├── run.sh                    # 统一运行脚本
├── report.tex               # 完整实验报告
├── README.md                # 本文件
├── 实训课1-基础题/
│   └── src/
│       ├── sourcefile.cpp   # CPU版本矩阵乘法
│       ├── sourcefile_dcu.cpp # DCU版本矩阵乘法
│       └── compile.sh       # 编译脚本
├── 实训课2-进阶题1/
│   └── src/
│       ├── sourcefile_mlp_cpu.cpp # CPU版本MLP
│       ├── sourcefile_mlp_forward.cpp # DCU前向传播
│       ├── sourcefile_mlp_optimized.cpp # DCU优化版本
│       └── compile.sh       # 编译脚本
└── 实训课3-进阶题2/
    ├── src/
    │   ├── compare_cpu.cpp  # CPU带宽预测
    │   ├── compare_cpu_full.cpp # CPU完整版本
    │   ├── mlp_dcu.cpp      # DCU带宽预测
    │   └── compile.sh       # 编译脚本
    └── data/
        └── bandwidth_data.json # 卫星网络数据
```

## 关键发现

1. **大规模计算**：DCU 具有压倒性优势（1800+倍加速）
2. **中等规模计算**：DCU 仍有明显优势，需要适当优化
3. **小规模复杂任务**：CPU 的单线程性能和灵活性更适合

## 技术特点

- 完整的 CPU 到 DCU 移植和优化流程
- 多种并行计算技术对比（OpenMP、MPI、分块优化、异构计算）
- 真实应用场景验证（Starlink 卫星网络数据）
- 详细的性能分析和优化建议

## 作者

仇科文 (2312237)  
南开大学计算机科学与技术专业
