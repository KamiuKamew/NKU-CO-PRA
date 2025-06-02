#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <chrono>

// 编译执行方式参考：
// 编译
// g++ -fopenmp -o matrix_mul matrix_mul.cpp

// 运行 baseline
// ./matrix_mul baseline

// 运行 OpenMP
// ./matrix_mul openmp

// 运行 子块并行优化
// ./matrix_mul block

// 运行 混合优化方法
// ./matrix_mul other

// 初始化矩阵（以一维数组形式表示），用于随机填充浮点数
void init_matrix(std::vector<double> &mat, int rows, int cols)
{
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-100.0, 100.0);
    for (int i = 0; i < rows * cols; ++i)
        mat[i] = dist(gen);
}

// 验证计算优化后的矩阵计算和baseline实现是否结果一致，可以设计其他验证方法，来验证计算的正确性和性能
bool validate(const std::vector<double> &A, const std::vector<double> &B, int rows, int cols, double tol = 1e-6)
{
    for (int i = 0; i < rows * cols; ++i)
        if (std::abs(A[i] - B[i]) > tol)
            return false;
    return true;
}

// 基础的矩阵乘法baseline实现（使用一维数组）
void matmul_baseline(const std::vector<double> &A,
                     const std::vector<double> &B,
                     std::vector<double> &C, int N, int M, int P)
{
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < P; ++j)
        {
            C[i * P + j] = 0;
            for (int k = 0; k < M; ++k)
                C[i * P + j] += A[i * M + k] * B[k * P + j];
        }
}

// 方式1: 利用OpenMP进行多线程并发的编程
void matmul_openmp(const std::vector<double> &A,
                   const std::vector<double> &B,
                   std::vector<double> &C, int N, int M, int P)
{
    #pragma omp parallel for
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < P; ++j)
        {
            C[i * P + j] = 0;
            for (int k = 0; k < M; ++k)
                C[i * P + j] += A[i * M + k] * B[k * P + j];
        }
}

// 方式2: 利用子块并行思想，进行缓存友好型的并行优化方法
void matmul_block_tiling(const std::vector<double> &A,
                         const std::vector<double> &B,
                         std::vector<double> &C, int N, int M, int P, int block_size = 32)
{
    // 初始化结果矩阵为0
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < P; ++j)
            C[i * P + j] = 0;
    
    // 分块计算
    for (int ii = 0; ii < N; ii += block_size)
        for (int jj = 0; jj < P; jj += block_size)
            for (int kk = 0; kk < M; kk += block_size)
                // 计算当前块
                for (int i = ii; i < std::min(ii + block_size, N); ++i)
                    for (int j = jj; j < std::min(jj + block_size, P); ++j)
                    {
                        double sum = 0;
                        for (int k = kk; k < std::min(kk + block_size, M); ++k)
                            sum += A[i * M + k] * B[k * P + j];
                        C[i * P + j] += sum;
                    }
}

// 方式4: 其他方式 - 结合OpenMP和子块优化的混合优化方法
void matmul_other(const std::vector<double> &A,
                  const std::vector<double> &B,
                  std::vector<double> &C, int N, int M, int P)
{
    // 初始化结果矩阵为0
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < P; ++j)
            C[i * P + j] = 0;
    
    const int block_size = 32;
    
    // 结合OpenMP和子块优化
    #pragma omp parallel for collapse(2)
    for (int ii = 0; ii < N; ii += block_size)
        for (int jj = 0; jj < P; jj += block_size)
            for (int kk = 0; kk < M; kk += block_size)
                // 计算当前块
                for (int i = ii; i < std::min(ii + block_size, N); ++i)
                    for (int j = jj; j < std::min(jj + block_size, P); ++j)
                    {
                        double sum = 0;
                        for (int k = kk; k < std::min(kk + block_size, M); ++k)
                            sum += A[i * M + k] * B[k * P + j];
                        
                        #pragma omp atomic
                        C[i * P + j] += sum;
                    }
}

int main(int argc, char **argv)
{
    // 调整矩阵规模，以便在有限资源环境中快速测试
    const int N = 512, M = 1024, P = 256;
    std::string mode = argc >= 2 ? argv[1] : "baseline";

    std::vector<double> A(N * M);
    std::vector<double> B(M * P);
    std::vector<double> C(N * P, 0);
    std::vector<double> C_ref(N * P, 0);

    std::cout << "Initializing matrices..." << std::endl;
    init_matrix(A, N, M);
    init_matrix(B, M, P);
    
    std::cout << "Matrix A: " << N << "x" << M << std::endl;
    std::cout << "Matrix B: " << M << "x" << P << std::endl;
    std::cout << "Matrix C: " << N << "x" << P << std::endl;
    
    // 测量baseline性能
    std::cout << "Running baseline implementation..." << std::endl;
    auto baseline_start = std::chrono::high_resolution_clock::now();
    matmul_baseline(A, B, C_ref, N, M, P);
    auto baseline_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> baseline_elapsed = baseline_end - baseline_start;

    if (mode == "baseline")
    {
        std::cout << "[Baseline] Time: " << baseline_elapsed.count() << " seconds" << std::endl;
    }
    else if (mode == "openmp")
    {
        std::cout << "Running OpenMP implementation..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        matmul_openmp(A, B, C, N, M, P);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        
        std::cout << "[OpenMP] Valid: " << validate(C, C_ref, N, P) << std::endl;
        std::cout << "[OpenMP] Time: " << elapsed.count() << " seconds" << std::endl;
        std::cout << "[Baseline] Time: " << baseline_elapsed.count() << " seconds" << std::endl;
        std::cout << "[OpenMP] Speedup: " << baseline_elapsed.count() / elapsed.count() << "x" << std::endl;
    }
    else if (mode == "block")
    {
        std::cout << "Running Block Tiling implementation..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        matmul_block_tiling(A, B, C, N, M, P);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        
        std::cout << "[Block Parallel] Valid: " << validate(C, C_ref, N, P) << std::endl;
        std::cout << "[Block Parallel] Time: " << elapsed.count() << " seconds" << std::endl;
        std::cout << "[Baseline] Time: " << baseline_elapsed.count() << " seconds" << std::endl;
        std::cout << "[Block Parallel] Speedup: " << baseline_elapsed.count() / elapsed.count() << "x" << std::endl;
    }
    else if (mode == "other")
    {
        std::cout << "Running Hybrid OpenMP+Block implementation..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        matmul_other(A, B, C, N, M, P);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        
        std::cout << "[Hybrid OpenMP+Block] Valid: " << validate(C, C_ref, N, P) << std::endl;
        std::cout << "[Hybrid OpenMP+Block] Time: " << elapsed.count() << " seconds" << std::endl;
        std::cout << "[Baseline] Time: " << baseline_elapsed.count() << " seconds" << std::endl;
        std::cout << "[Hybrid OpenMP+Block] Speedup: " << baseline_elapsed.count() / elapsed.count() << "x" << std::endl;
    }
    else
    {
        std::cerr << "Usage: ./matrix_mul [baseline|openmp|block|other]" << std::endl;
    }
    
    return 0;
}
