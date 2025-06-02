#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <chrono>

// 编译文件
// hipcc sourcefile_mlp_optimized.cpp -o mlp_optimized

#define BATCH 1024
#define I 10
#define H 20
#define O 5
#define BLOCK_SIZE 16
#define TILE_SIZE 32

// 优化版矩阵乘法内核 - 使用共享内存和tiling
__global__ void matmul_optimized_kernel(const double *A, const double *B, double *C, int M, int N, int K)
{
    __shared__ double As[TILE_SIZE][TILE_SIZE];
    __shared__ double Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    double sum = 0.0;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t)
    {
        // 加载数据到共享内存
        if (row < M && t * TILE_SIZE + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0;

        if (col < N && t * TILE_SIZE + threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();

        // 计算部分和
        for (int k = 0; k < TILE_SIZE; ++k)
        {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N)
    {
        C[row * N + col] = sum;
    }
}

// 融合的偏置加法和ReLU内核
__global__ void add_bias_relu_kernel(double *C, const double *bias, int M, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N)
    {
        double val = C[row * N + col] + bias[col];
        C[row * N + col] = fmax(0.0, val); // ReLU
    }
}

// 仅偏置加法内核
__global__ void add_bias_kernel(double *C, const double *bias, int M, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N)
    {
        C[row * N + col] += bias[col];
    }
}

// 向量化的初始化内核
__global__ void init_kernel(double *data, int size, double value)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size; i += stride)
    {
        data[i] = value;
    }
}

// 随机初始化
void random_init(std::vector<double> &mat)
{
    for (auto &val : mat)
    {
        val = static_cast<double>(rand()) / RAND_MAX * 2 - 1; // [-1, 1]
    }
}

// CPU版本的MLP前向传播（用于验证）
void mlp_forward_cpu(const std::vector<double> &X, const std::vector<double> &W1, const std::vector<double> &B1,
                     const std::vector<double> &W2, const std::vector<double> &B2,
                     std::vector<double> &Y)
{
    std::vector<double> H(BATCH * H);

    // 第一层：H = X * W1 + B1
    for (int i = 0; i < BATCH; ++i)
    {
        for (int j = 0; j < H; ++j)
        {
            double sum = 0.0;
            for (int k = 0; k < I; ++k)
            {
                sum += X[i * I + k] * W1[k * H + j];
            }
            H[i * H + j] = sum + B1[j];
            // ReLU
            H[i * H + j] = std::max(0.0, H[i * H + j]);
        }
    }

    // 第二层：Y = H * W2 + B2
    for (int i = 0; i < BATCH; ++i)
    {
        for (int j = 0; j < O; ++j)
        {
            double sum = 0.0;
            for (int k = 0; k < H; ++k)
            {
                sum += H[i * H + k] * W2[k * O + j];
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

class MLPOptimized
{
private:
    double *d_X, *d_W1, *d_B1, *d_H, *d_W2, *d_B2, *d_Y;
    hipStream_t stream;

public:
    MLPOptimized()
    {
        // 分配设备内存
        hipMalloc(&d_X, BATCH * I * sizeof(double));
        hipMalloc(&d_W1, I * H * sizeof(double));
        hipMalloc(&d_B1, H * sizeof(double));
        hipMalloc(&d_H, BATCH * H * sizeof(double));
        hipMalloc(&d_W2, H * O * sizeof(double));
        hipMalloc(&d_B2, O * sizeof(double));
        hipMalloc(&d_Y, BATCH * O * sizeof(double));

        // 创建流
        hipStreamCreate(&stream);
    }

    ~MLPOptimized()
    {
        hipFree(d_X);
        hipFree(d_W1);
        hipFree(d_B1);
        hipFree(d_H);
        hipFree(d_W2);
        hipFree(d_B2);
        hipFree(d_Y);
        hipStreamDestroy(stream);
    }

    void forward(const std::vector<double> &h_X, const std::vector<double> &h_W1, const std::vector<double> &h_B1,
                 const std::vector<double> &h_W2, const std::vector<double> &h_B2,
                 std::vector<double> &h_Y)
    {
        // 异步拷贝数据到设备
        hipMemcpyAsync(d_X, h_X.data(), BATCH * I * sizeof(double), hipMemcpyHostToDevice, stream);
        hipMemcpyAsync(d_W1, h_W1.data(), I * H * sizeof(double), hipMemcpyHostToDevice, stream);
        hipMemcpyAsync(d_B1, h_B1.data(), H * sizeof(double), hipMemcpyHostToDevice, stream);
        hipMemcpyAsync(d_W2, h_W2.data(), H * O * sizeof(double), hipMemcpyHostToDevice, stream);
        hipMemcpyAsync(d_B2, h_B2.data(), O * sizeof(double), hipMemcpyHostToDevice, stream);

        // 第一层：隐藏层计算 H = X * W1 + B1，然后ReLU
        dim3 block1(TILE_SIZE, TILE_SIZE);
        dim3 grid1((H + TILE_SIZE - 1) / TILE_SIZE, (BATCH + TILE_SIZE - 1) / TILE_SIZE);

        hipLaunchKernelGGL(matmul_optimized_kernel, grid1, block1, 0, stream, d_X, d_W1, d_H, BATCH, H, I);

        dim3 block_bias(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid_bias((H + BLOCK_SIZE - 1) / BLOCK_SIZE, (BATCH + BLOCK_SIZE - 1) / BLOCK_SIZE);
        hipLaunchKernelGGL(add_bias_relu_kernel, grid_bias, block_bias, 0, stream, d_H, d_B1, BATCH, H);

        // 第二层：输出层计算 Y = H * W2 + B2
        dim3 grid2((O + TILE_SIZE - 1) / TILE_SIZE, (BATCH + TILE_SIZE - 1) / TILE_SIZE);

        hipLaunchKernelGGL(matmul_optimized_kernel, grid2, block1, 0, stream, d_H, d_W2, d_Y, BATCH, O, H);

        dim3 grid_bias2((O + BLOCK_SIZE - 1) / BLOCK_SIZE, (BATCH + BLOCK_SIZE - 1) / BLOCK_SIZE);
        hipLaunchKernelGGL(add_bias_kernel, grid_bias2, block_bias, 0, stream, d_Y, d_B2, BATCH, O);

        // 异步拷贝结果回主机
        hipMemcpyAsync(h_Y.data(), d_Y, BATCH * O * sizeof(double), hipMemcpyDeviceToHost, stream);
        hipStreamSynchronize(stream);
    }
};

int main()
{
    srand(42); // 固定随机种子确保可重现性

    // 主机端数据
    std::vector<double> h_X(BATCH * I), h_W1(I * H), h_B1(H), h_W2(H * O), h_B2(O);
    std::vector<double> h_Y(BATCH * O), h_Y_cpu(BATCH * O);

    // 初始化数据
    random_init(h_X);
    random_init(h_W1);
    random_init(h_B1);
    random_init(h_W2);
    random_init(h_B2);

    std::cout << "MLP Forward Propagation - Optimized DCU Implementation" << std::endl;
    std::cout << "Network: " << BATCH << "×" << I << " → " << I << "×" << H << " (ReLU) → " << H << "×" << O << std::endl;
    std::cout << "Optimizations: Shared Memory + Tiling + Kernel Fusion + Async Memory" << std::endl;

    // CPU基准计算
    auto cpu_start = std::chrono::high_resolution_clock::now();
    mlp_forward_cpu(h_X, h_W1, h_B1, h_W2, h_B2, h_Y_cpu);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count() / 1000.0;

    // 优化的DCU实现
    MLPOptimized mlp;

    // 预热
    mlp.forward(h_X, h_W1, h_B1, h_W2, h_B2, h_Y);

    auto gpu_start = std::chrono::high_resolution_clock::now();
    mlp.forward(h_X, h_W1, h_B1, h_W2, h_B2, h_Y);
    auto gpu_end = std::chrono::high_resolution_clock::now();
    auto gpu_time = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_start).count() / 1000.0;

    // 性能统计
    std::cout << "\n=== Optimized Performance Results ===" << std::endl;
    std::cout << "CPU Time: " << cpu_time << " ms" << std::endl;
    std::cout << "DCU Time (Optimized): " << gpu_time << " ms" << std::endl;
    std::cout << "Speedup: " << cpu_time / gpu_time << "x" << std::endl;

    // 验证结果
    if (validate_results(h_Y_cpu, h_Y))
    {
        std::cout << "\n✓ Validation PASSED: Optimized DCU results match CPU baseline" << std::endl;
    }
    else
    {
        std::cout << "\n✗ Validation FAILED: Optimized DCU results differ from CPU baseline" << std::endl;
    }

    // 打印部分输出结果
    std::cout << "\n=== Sample Outputs ===" << std::endl;
    for (int i = 0; i < 3; ++i)
    {
        std::cout << "Batch[" << i << "]: ";
        for (int j = 0; j < O; ++j)
        {
            std::cout << h_Y[i * O + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}