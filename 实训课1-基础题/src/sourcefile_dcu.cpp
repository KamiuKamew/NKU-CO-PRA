#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>

// 编译
// hipcc sourcefile_dcu.cpp -o outputfile_dcu
// 执行
// ./outputfile_dcu

#define N 1024
#define M 2024
#define P 512
#define BLOCK_SIZE 16

// 主要修改函数
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

void init_matrix(std::vector<double> &mat)
{
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (auto &x : mat)
        x = dist(gen);
    return;
}

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

bool validate(const std::vector<double> &ref, const std::vector<double> &test)
{
    for (size_t i = 0; i < ref.size(); ++i)
        if (std::abs(ref[i] - test[i]) > 1e-6)
            return false;
    return true;
}

int main()
{
    std::vector<double> A(N * M), B(M * P), C(N * P), C_ref(N * P);
    init_matrix(A);
    init_matrix(B);

    // CPU baseline
    matmul_cpu(A, B, C_ref);

    // 主要修改部分
    // Allocate and copy to device, use matmul_kernel to compute in DCU
    double *d_A, *d_B, *d_C;

    // 分配设备内存
    hipMalloc(&d_A, N * M * sizeof(double));
    hipMalloc(&d_B, M * P * sizeof(double));
    hipMalloc(&d_C, N * P * sizeof(double));

    // 将数据从主机复制到设备
    hipMemcpy(d_A, A.data(), N * M * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_B, B.data(), M * P * sizeof(double), hipMemcpyHostToDevice);

    // 设置线程块和网格大小
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((P + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // 启动内核
    hipLaunchKernelGGL(matmul_kernel, gridDim, blockDim, 0, 0, d_A, d_B, d_C, N, M, P);

    // 将结果从设备复制回主机
    hipMemcpy(C.data(), d_C, N * P * sizeof(double), hipMemcpyDeviceToHost);

    if (validate(C_ref, C))
    {
        std::cout << "[HIP] Valid: 1" << std::endl;
    }
    else
    {
        std::cout << "[HIP] Valid: 0" << std::endl;
    }

    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);

    return 0;
}
