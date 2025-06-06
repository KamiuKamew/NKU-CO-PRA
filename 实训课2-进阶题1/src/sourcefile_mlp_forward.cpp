#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <chrono>

// 编译文件
// hipcc sourcefile_mlp_forward.cpp -o mlp_forward
// 执行文件
// ./mlp_forward 或者 hipprof ./mlp_forward

#define BATCH 1024
#define I 10
#define H_SIZE 20
#define O 5
#define BLOCK_SIZE 16

// 矩阵乘法内核
__global__ void matmul_kernel(const double *A, const double *B, double *C, int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N)
    {
        double sum = 0.0;
        for (int k = 0; k < K; ++k)
        {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// 偏置加法内核
__global__ void add_bias_kernel(double *C, const double *bias, int M, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N)
    {
        C[row * N + col] += bias[col];
    }
}

// ReLU激活函数内核
__global__ void relu_kernel(double *A, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        A[idx] = fmax(0.0, A[idx]);
    }
}

// 随机初始化
void random_init(std::vector<double> &mat)
{
    for (auto &val : mat)
    {
        val = static_cast<double>(rand()) / RAND_MAX * 2 - 1;
    }
}

// CPU版本MLP前向传播
void mlp_forward_cpu(const std::vector<double> &X, const std::vector<double> &W1, const std::vector<double> &B1,
                     const std::vector<double> &W2, const std::vector<double> &B2,
                     std::vector<double> &Y)
{
    std::vector<double> H(BATCH * H_SIZE);

    for (int i = 0; i < BATCH; ++i)
    {
        for (int j = 0; j < H_SIZE; ++j)
        {
            double sum = 0.0;
            for (int k = 0; k < I; ++k)
            {
                sum += X[i * I + k] * W1[k * H_SIZE + j];
            }
            H[i * H_SIZE + j] = sum + B1[j];
            H[i * H_SIZE + j] = std::max(0.0, H[i * H_SIZE + j]);
        }
    }

    for (int i = 0; i < BATCH; ++i)
    {
        for (int j = 0; j < O; ++j)
        {
            double sum = 0.0;
            for (int k = 0; k < H_SIZE; ++k)
            {
                sum += H[i * H_SIZE + k] * W2[k * O + j];
            }
            Y[i * O + j] = sum + B2[j];
        }
    }
}

// 验证结果
bool validate_results(const std::vector<double> &cpu_result, const std::vector<double> &gpu_result, double tol = 1e-6)
{
    for (size_t i = 0; i < cpu_result.size(); ++i)
    {
        if (std::abs(cpu_result[i] - gpu_result[i]) > tol)
        {
            std::cout << "Validation failed at index " << i
                      << ": CPU=" << cpu_result[i]
                      << ", GPU=" << gpu_result[i]
                      << ", diff=" << std::abs(cpu_result[i] - gpu_result[i]) << std::endl;
            return false;
        }
    }
    return true;
}

int main()
{
    srand(42);

    std::vector<double> h_X(BATCH * I), h_W1(I * H_SIZE), h_B1(H_SIZE), h_W2(H_SIZE * O), h_B2(O);
    std::vector<double> h_H(BATCH * H_SIZE), h_Y(BATCH * O);
    std::vector<double> h_Y_cpu(BATCH * O);

    random_init(h_X);
    random_init(h_W1);
    random_init(h_B1);
    random_init(h_W2);
    random_init(h_B2);

    std::cout << "MLP Forward Propagation - DCU Implementation" << std::endl;
    std::cout << "Network: " << BATCH << "×" << I << " → " << I << "×" << H_SIZE << " (ReLU) → " << H_SIZE << "×" << O << std::endl;

    auto cpu_start = std::chrono::high_resolution_clock::now();
    mlp_forward_cpu(h_X, h_W1, h_B1, h_W2, h_B2, h_Y_cpu);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count() / 1000.0;

    double *d_X, *d_W1, *d_B1, *d_H, *d_W2, *d_B2, *d_Y;

    hipMalloc(&d_X, BATCH * I * sizeof(double));
    hipMalloc(&d_W1, I * H_SIZE * sizeof(double));
    hipMalloc(&d_B1, H_SIZE * sizeof(double));
    hipMalloc(&d_H, BATCH * H_SIZE * sizeof(double));
    hipMalloc(&d_W2, H_SIZE * O * sizeof(double));
    hipMalloc(&d_B2, O * sizeof(double));
    hipMalloc(&d_Y, BATCH * O * sizeof(double));

    hipMemcpy(d_X, h_X.data(), BATCH * I * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_W1, h_W1.data(), I * H_SIZE * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_B1, h_B1.data(), H_SIZE * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_W2, h_W2.data(), H_SIZE * O * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_B2, h_B2.data(), O * sizeof(double), hipMemcpyHostToDevice);

    auto gpu_start = std::chrono::high_resolution_clock::now();

    dim3 block1(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid1((H_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, (BATCH + BLOCK_SIZE - 1) / BLOCK_SIZE);

    hipLaunchKernelGGL(matmul_kernel, grid1, block1, 0, 0, d_X, d_W1, d_H, BATCH, H_SIZE, I);
    hipLaunchKernelGGL(add_bias_kernel, grid1, block1, 0, 0, d_H, d_B1, BATCH, H_SIZE);

    int threads_relu1 = (BATCH * H_SIZE + 255) / 256;
    hipLaunchKernelGGL(relu_kernel, dim3(threads_relu1), dim3(256), 0, 0, d_H, BATCH * H_SIZE);

    dim3 grid2((O + BLOCK_SIZE - 1) / BLOCK_SIZE, (BATCH + BLOCK_SIZE - 1) / BLOCK_SIZE);

    hipLaunchKernelGGL(matmul_kernel, grid2, block1, 0, 0, d_H, d_W2, d_Y, BATCH, O, H_SIZE);
    hipLaunchKernelGGL(add_bias_kernel, grid2, block1, 0, 0, d_Y, d_B2, BATCH, O);

    hipDeviceSynchronize();
    auto gpu_end = std::chrono::high_resolution_clock::now();
    auto gpu_time = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_start).count() / 1000.0;

    hipMemcpy(h_Y.data(), d_Y, BATCH * O * sizeof(double), hipMemcpyDeviceToHost);

    std::cout << "\n=== Performance Results ===" << std::endl;
    std::cout << "CPU Time: " << cpu_time << " ms" << std::endl;
    std::cout << "DCU Time: " << gpu_time << " ms" << std::endl;
    std::cout << "Speedup: " << (double)cpu_time / gpu_time << "x" << std::endl;

    if (validate_results(h_Y_cpu, h_Y))
    {
        std::cout << "\n✓ Validation PASSED: DCU results match CPU baseline" << std::endl;
    }
    else
    {
        std::cout << "\n✗ Validation FAILED: Results differ between CPU and DCU" << std::endl;
    }

    hipFree(d_X);
    hipFree(d_W1);
    hipFree(d_B1);
    hipFree(d_H);
    hipFree(d_W2);
    hipFree(d_B2);
    hipFree(d_Y);

    return 0;
}
