#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <fstream>
#include <iomanip>

// 编译
// hipcc sourcefile_dcu.cpp -o outputfile_dcu
// 执行
// ./outputfile_dcu

#define N 1024
#define M 2024
#define P 512
#define BLOCK_SIZE 16

// DCU矩阵乘法内核
__global__ void matmul_kernel(const double *A, const double *B, double *C, int n, int m, int p)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < p)
    {
        double sum = 0.0;
        for (int k = 0; k < m; ++k)
        {
            sum += A[row * m + k] * B[k * p + col];
        }
        C[row * p + col] = sum;
    }
}

// 初始化矩阵
void init_matrix(std::vector<double> &mat)
{
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (auto &x : mat)
        x = dist(gen);
    return;
}

// CPU基准矩阵乘法
void matmul_cpu(const std::vector<double> &A, const std::vector<double> &B, std::vector<double> &C)
{
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < P; ++j)
        {
            double sum = 0.0;
            for (int k = 0; k < M; ++k)
                sum += A[i * M + k] * B[k * P + j];
            C[i * P + j] = sum;
        }
    return;
}

// 验证结果
bool validate(const std::vector<double> &ref, const std::vector<double> &test)
{
    for (size_t i = 0; i < ref.size(); ++i)
        if (std::abs(ref[i] - test[i]) > 1e-6)
            return false;
    return true;
}

// 性能结果保存
void benchmark_and_save(const std::string &method, double time_ms)
{
    std::ofstream outfile("performance_results.txt", std::ios::app);
    outfile << method << "," << time_ms << std::endl;
}

int main()
{
    const int NUM_RUNS = 5;
    double total_time_cpu = 0.0;
    double total_time_dcu = 0.0;

    std::vector<double> A(N * M), B(M * P), C(N * P), C_ref(N * P);
    init_matrix(A);
    init_matrix(B);

    for (int run = 0; run < NUM_RUNS; ++run)
    {
        auto start = std::chrono::high_resolution_clock::now();
        matmul_cpu(A, B, C_ref);
        auto end = std::chrono::high_resolution_clock::now();
        total_time_cpu += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    }
    double avg_time_cpu = total_time_cpu / NUM_RUNS;
    std::cout << "[CPU] Average time: " << avg_time_cpu << " ms" << std::endl;
    benchmark_and_save("CPU", avg_time_cpu);

    double *d_A, *d_B, *d_C;

    hipMalloc(&d_A, N * M * sizeof(double));
    hipMalloc(&d_B, M * P * sizeof(double));
    hipMalloc(&d_C, N * P * sizeof(double));

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((P + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    for (int run = 0; run < NUM_RUNS; ++run)
    {
        auto start = std::chrono::high_resolution_clock::now();

        hipMemcpy(d_A, A.data(), N * M * sizeof(double), hipMemcpyHostToDevice);
        hipMemcpy(d_B, B.data(), M * P * sizeof(double), hipMemcpyHostToDevice);

        hipLaunchKernelGGL(matmul_kernel, gridDim, blockDim, 0, 0, d_A, d_B, d_C, N, M, P);

        hipDeviceSynchronize();

        hipMemcpy(C.data(), d_C, N * P * sizeof(double), hipMemcpyDeviceToHost);

        auto end = std::chrono::high_resolution_clock::now();
        total_time_dcu += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    }

    double avg_time_dcu = total_time_dcu / NUM_RUNS;
    std::cout << "[DCU] Average time: " << avg_time_dcu << " ms" << std::endl;
    std::cout << "[DCU] Speedup over CPU: " << avg_time_cpu / avg_time_dcu << "x" << std::endl;
    benchmark_and_save("DCU", avg_time_dcu);

    if (validate(C_ref, C))
    {
        std::cout << "[DCU] Valid: 1" << std::endl;
    }
    else
    {
        std::cout << "[DCU] Valid: 0" << std::endl;
    }

    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);

    return 0;
}
