#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <mpi.h>
#include <chrono>
#include <fstream>
#include <iomanip>

// 编译执行方式参考：
// 编译， 也可以使用g++，但使用MPI时需使用mpic
// mpic++ -fopenmp -o outputfile sourcefile.cpp

// 运行 baseline
// ./outputfile baseline

// 运行 OpenMP
// ./outputfile openmp

// 运行 子块并行优化
// ./outputfile block

// 运行 MPI（假设 4 个进程）
// mpirun -np 4 ./outputfile mpi

// 运行 MPI（假设 4 个进程）
// ./outputfile other

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
    int error_count = 0;
    double max_error = 0.0;
    int first_error_idx = -1;

    for (int i = 0; i < rows * cols; ++i)
    {
        double error = std::abs(A[i] - B[i]);
        if (error > tol)
        {
            error_count++;
            if (error > max_error)
            {
                max_error = error;
            }
            if (first_error_idx == -1)
            {
                first_error_idx = i;
            }
        }
    }

    if (error_count > 0)
    {
        std::cout << "[DEBUG] Validation failed:" << std::endl;
        std::cout << "[DEBUG] Total errors: " << error_count << " out of " << rows * cols << std::endl;
        std::cout << "[DEBUG] Max error: " << max_error << std::endl;
        std::cout << "[DEBUG] First error at index " << first_error_idx
                  << ": expected " << B[first_error_idx]
                  << ", got " << A[first_error_idx] << std::endl;
        return false;
    }
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

// 方式1: 利用OpenMP进行多线程并发的编程 （主要修改函数）
void matmul_openmp(const std::vector<double> &A,
                   const std::vector<double> &B,
                   std::vector<double> &C, int N, int M, int P)
{
    std::fill(C.begin(), C.end(), 0.0);
#pragma omp parallel for collapse(2)
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < P; ++j)
        {
            double sum = 0.0;
            for (int k = 0; k < M; ++k)
            {
                sum += A[i * M + k] * B[k * P + j];
            }
            C[i * P + j] = sum;
        }
    }
}

// 方式2: 利用子块并行思想，进行缓存友好型的并行优化方法 （主要修改函数)
void matmul_block_tiling(const std::vector<double> &A,
                         const std::vector<double> &B,
                         std::vector<double> &C, int N, int M, int P, int block_size)
{
    std::fill(C.begin(), C.end(), 0.0);

#pragma omp parallel for collapse(3)
    for (int i = 0; i < N; i += block_size)
    {
        for (int j = 0; j < P; j += block_size)
        {
            for (int k = 0; k < M; k += block_size)
            {
                // 计算当前块的实际大小
                int i_end = std::min(i + block_size, N);
                int j_end = std::min(j + block_size, P);
                int k_end = std::min(k + block_size, M);

                // 对块内元素进行计算
                for (int ii = i; ii < i_end; ++ii)
                {
                    for (int jj = j; jj < j_end; ++jj)
                    {
                        double sum = C[ii * P + jj]; // 累加到现有结果上
                        for (int kk = k; kk < k_end; ++kk)
                        {
                            sum += A[ii * M + kk] * B[kk * P + jj];
                        }
                        C[ii * P + jj] = sum;
                    }
                }
            }
        }
    }
}

// 方式3: 利用MPI消息传递，实现多进程并行优化 （主要修改函数）
void matmul_mpi(int N, int M, int P)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 计算每个进程需要处理的行数
    int rows_per_proc = N / size;
    int extra_rows = N % size;
    int my_rows = rows_per_proc + (rank < extra_rows ? 1 : 0);
    int my_start = rank * rows_per_proc + std::min(rank, extra_rows);

    // 分配局部矩阵内存
    std::vector<double> A_local(my_rows * M);
    std::vector<double> B(M * P);
    std::vector<double> C_local(my_rows * P, 0);
    std::vector<double> A, C;

    // 在根进程初始化矩阵
    if (rank == 0)
    {
        A.resize(N * M);
        C.resize(N * P);
        init_matrix(A, N, M);
        init_matrix(B, M, P);
    }

    // 广播矩阵B到所有进程
    MPI_Bcast(B.data(), M * P, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 分发矩阵A的行到各个进程
    std::vector<int> sendcounts(size);
    std::vector<int> displs(size);
    for (int i = 0; i < size; i++)
    {
        sendcounts[i] = (rows_per_proc + (i < extra_rows ? 1 : 0)) * M;
        displs[i] = (i * rows_per_proc + std::min(i, extra_rows)) * M;
    }

    MPI_Scatterv(rank == 0 ? A.data() : nullptr, sendcounts.data(), displs.data(),
                 MPI_DOUBLE, A_local.data(), my_rows * M, MPI_DOUBLE, 0, MPI_COMM_WORLD);

// 计算局部矩阵乘法
#pragma omp parallel for collapse(2)
    for (int i = 0; i < my_rows; ++i)
    {
        for (int j = 0; j < P; ++j)
        {
            double sum = 0.0;
            for (int k = 0; k < M; ++k)
            {
                sum += A_local[i * M + k] * B[k * P + j];
            }
            C_local[i * P + j] = sum;
        }
    }

    // 收集结果到根进程
    std::vector<int> recvcounts(size);
    std::vector<int> rdispls(size);
    for (int i = 0; i < size; i++)
    {
        recvcounts[i] = (rows_per_proc + (i < extra_rows ? 1 : 0)) * P;
        rdispls[i] = (i * rows_per_proc + std::min(i, extra_rows)) * P;
    }

    MPI_Gatherv(C_local.data(), my_rows * P, MPI_DOUBLE,
                rank == 0 ? C.data() : nullptr, recvcounts.data(), rdispls.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 在根进程验证结果
    if (rank == 0)
    {
        std::vector<double> C_ref(N * P);
        matmul_baseline(A, B, C_ref, N, M, P);
        std::cout << "[MPI] Valid: " << validate(C, C_ref, N, P) << std::endl;
    }
}

// 方式4: 其他方式 （主要修改函数）
void matmul_other(const std::vector<double> &A,
                  const std::vector<double> &B,
                  std::vector<double> &C, int N, int M, int P)
{
    std::cout << "Other methods..." << std::endl;
}

// 性能测试函数
void benchmark_and_save(const std::string &method, double time_ms)
{
    std::ofstream outfile("performance_results.txt", std::ios::app);
    outfile << method << "," << time_ms << std::endl;
}

int main(int argc, char **argv)
{
    const int N = 1024, M = 2048, P = 512;
    std::string mode = argc >= 2 ? argv[1] : "baseline";
    const int NUM_RUNS = 5; // 每种方法运行5次取平均

    if (mode == "mpi")
    {
        MPI_Init(&argc, &argv);

        auto start = std::chrono::high_resolution_clock::now();
        matmul_mpi(N, M, P);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0)
        {
            benchmark_and_save("MPI", duration);
        }

        MPI_Finalize();
        return 0;
    }

    std::vector<double> A(N * M);
    std::vector<double> B(M * P);
    std::vector<double> C(N * P, 0);
    std::vector<double> C_ref(N * P, 0);

    init_matrix(A, N, M);
    init_matrix(B, M, P);

    // 计算基准结果
    std::cout << "[DEBUG] Computing baseline reference..." << std::endl;
    matmul_baseline(A, B, C_ref, N, M, P);
    std::cout << "[DEBUG] Baseline reference computed. Sample values: "
              << C_ref[0] << ", " << C_ref[1] << ", " << C_ref[N * P - 1] << std::endl;

    double total_time = 0.0;

    for (int run = 0; run < NUM_RUNS; ++run)
    {
        if (mode == "baseline")
        {
            auto start = std::chrono::high_resolution_clock::now();
            matmul_baseline(A, B, C_ref, N, M, P);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            total_time += duration;

            if (run == NUM_RUNS - 1)
            {
                std::cout << "[Baseline] Average time: " << total_time / NUM_RUNS << " ms" << std::endl;
                benchmark_and_save("Baseline", total_time / NUM_RUNS);
            }
        }
        else if (mode == "openmp")
        {
            std::cout << "[DEBUG] OpenMP run " << run + 1 << "..." << std::endl;
            auto start = std::chrono::high_resolution_clock::now();
            matmul_openmp(A, B, C, N, M, P);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            total_time += duration;

            if (run == NUM_RUNS - 1)
            {
                std::cout << "[DEBUG] OpenMP result sample values: "
                          << C[0] << ", " << C[1] << ", " << C[N * P - 1] << std::endl;
                std::cout << "[OpenMP] Average time: " << total_time / NUM_RUNS << " ms" << std::endl;
                std::cout << "[OpenMP] Valid: " << validate(C, C_ref, N, P) << std::endl;
                benchmark_and_save("OpenMP", total_time / NUM_RUNS);
            }
        }
        else if (mode == "block")
        {
            std::cout << "[DEBUG] Block run " << run + 1 << "..." << std::endl;
            auto start = std::chrono::high_resolution_clock::now();
            matmul_block_tiling(A, B, C, N, M, P, 64);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            total_time += duration;

            if (run == NUM_RUNS - 1)
            {
                std::cout << "[DEBUG] Block result sample values: "
                          << C[0] << ", " << C[1] << ", " << C[N * P - 1] << std::endl;
                std::cout << "[Block] Average time: " << total_time / NUM_RUNS << " ms" << std::endl;
                std::cout << "[Block] Valid: " << validate(C, C_ref, N, P) << std::endl;
                benchmark_and_save("Block", total_time / NUM_RUNS);
            }
        }
        else if (mode == "other")
        {
            auto start = std::chrono::high_resolution_clock::now();
            matmul_other(A, B, C, N, M, P);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            total_time += duration;

            if (run == NUM_RUNS - 1)
            {
                std::cout << "[Other] Average time: " << total_time / NUM_RUNS << " ms" << std::endl;
                std::cout << "[Other] Valid: " << validate(C, C_ref, N, P) << std::endl;
                benchmark_and_save("Other", total_time / NUM_RUNS);
            }
        }
        else
        {
            std::cerr << "Usage: ./main [baseline|openmp|block|mpi]" << std::endl;
            return 1;
        }
    }
    return 0;
}
