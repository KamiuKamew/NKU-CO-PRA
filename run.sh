#!/bin/bash

# LEO卫星网络带宽预测系统统一运行脚本
# 用于编译和运行所有三个实训课题目

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# 检查系统依赖
check_dependencies() {
    print_header "检查系统依赖"
    
    if command -v mpic++ &> /dev/null; then
        print_success "MPI编译器已安装"
    else
        print_error "MPI编译器未安装"
        exit 1
    fi
    
    if command -v hipcc &> /dev/null; then
        print_success "HIP编译器已安装"
    else
        print_warning "HIP编译器未安装，将跳过DCU相关测试"
    fi
}

# 基础题：矩阵乘法优化
run_basic_task() {
    print_header "基础题：矩阵乘法优化"
    
    cd "实训课1-基础题/src"
    
    # 编译CPU版本
    print_success "编译CPU版本..."
    mpic++ -fopenmp -O3 -o matmul_cpu sourcefile.cpp
    
    # 编译DCU版本（如果可用）
    if command -v hipcc &> /dev/null; then
        print_success "编译DCU版本..."
        hipcc -O3 -o matmul_dcu sourcefile_dcu.cpp
        DCU_AVAILABLE=true
    else
        DCU_AVAILABLE=false
    fi
    
    # 运行基准测试
    rm -f performance_results.txt
    
    echo "运行矩阵乘法基准测试..."
    ./matmul_cpu baseline
    ./matmul_cpu openmp
    ./matmul_cpu block
    mpirun --allow-run-as-root -np 4 ./matmul_cpu mpi
    
    if [ "$DCU_AVAILABLE" = true ]; then
        ./matmul_dcu
    fi
    
    print_success "基础题完成"
    cd ../..
}

# 进阶题1：MLP前向传播
run_advanced_task1() {
    print_header "进阶题1：MLP前向传播"
    
    cd "实训课2-进阶题1/src"
    
    # 编译CPU版本
    print_success "编译CPU版本..."
    g++ -O3 -o mlp_cpu sourcefile_mlp_cpu.cpp
    
    # 编译DCU版本（如果可用）
    if command -v hipcc &> /dev/null; then
        print_success "编译DCU前向传播版本..."
        hipcc -O3 -o mlp_forward sourcefile_mlp_forward.cpp
        hipcc -O3 -o mlp_optimized sourcefile_mlp_optimized.cpp
        DCU_AVAILABLE=true
    else
        DCU_AVAILABLE=false
    fi
    
    # 运行测试
    echo "运行MLP前向传播测试..."
    ./mlp_cpu
    
    if [ "$DCU_AVAILABLE" = true ]; then
        ./mlp_forward
        ./mlp_optimized
    fi
    
    print_success "进阶题1完成"
    cd ../..
}

# 进阶题2：LEO卫星网络带宽预测
run_advanced_task2() {
    print_header "进阶题2：LEO卫星网络带宽预测"
    
    cd "实训课3-进阶题2/src"
    
    # 编译CPU版本
    print_success "编译CPU版本..."
    g++ -O3 -std=c++17 -o predict_cpu compare_cpu.cpp
    g++ -O3 -std=c++17 -o predict_cpu_full compare_cpu_full.cpp
    
    # 编译DCU版本（如果可用）
    if command -v hipcc &> /dev/null; then
        print_success "编译DCU版本..."
        hipcc -O3 -std=c++17 -o predict_dcu mlp_dcu.cpp
        DCU_AVAILABLE=true
    else
        DCU_AVAILABLE=false
    fi
    
    # 运行预测测试
    echo "运行LEO卫星网络带宽预测测试..."
    ./predict_cpu
    ./predict_cpu_full
    
    if [ "$DCU_AVAILABLE" = true ]; then
        ./predict_dcu
    fi
    
    print_success "进阶题2完成"
    cd ../..
}

# 生成性能报告
generate_report() {
    print_header "生成性能报告"
    
    echo "性能测试结果已保存到各个子目录中的performance_results.txt文件"
    echo "详细的实验报告请查看report.tex文件"
    
    print_success "所有任务完成"
}

# 主函数
main() {
    print_header "LEO卫星网络带宽预测系统性能优化"
    echo "南开大学计算机学院 智能计算创新设计赛（先导杯）"
    echo ""
    
    case "${1:-all}" in
        "basic")
            check_dependencies
            run_basic_task
            ;;
        "advanced1")
            check_dependencies
            run_advanced_task1
            ;;
        "advanced2")
            check_dependencies
            run_advanced_task2
            ;;
        "all")
            check_dependencies
            run_basic_task
            run_advanced_task1
            run_advanced_task2
            generate_report
            ;;
        "help")
            echo "用法: $0 [basic|advanced1|advanced2|all|help]"
            echo ""
            echo "  basic     - 运行基础题（矩阵乘法优化）"
            echo "  advanced1 - 运行进阶题1（MLP前向传播）"
            echo "  advanced2 - 运行进阶题2（LEO卫星网络带宽预测）"
            echo "  all       - 运行所有题目（默认）"
            echo "  help      - 显示此帮助信息"
            ;;
        *)
            print_error "未知参数: $1"
            echo "使用 '$0 help' 查看帮助信息"
            exit 1
            ;;
    esac
}

main "$@" 