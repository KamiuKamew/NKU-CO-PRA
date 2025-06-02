#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <random>

#define BATCH 1024
#define I 10
#define H_SIZE 20
#define O 5

// CPU版本的矩阵乘法
void matmul_cpu(const std::vector<double> &A, const std::vector<double> &B,
                std::vector<double> &C, int M, int N, int K)
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            double sum = 0.0;
            for (int k = 0; k < K; ++k)
            {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// CPU版本的偏置加法
void add_bias_cpu(std::vector<double> &C, const std::vector<double> &bias, int M, int N)
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            C[i * N + j] += bias[j];
        }
    }
}

// CPU版本的ReLU激活
void relu_cpu(std::vector<double> &A, int size)
{
    for (int i = 0; i < size; ++i)
    {
        A[i] = std::max(0.0, A[i]);
    }
}

// 随机初始化
void random_init(std::vector<double> &mat)
{
    std::random_device rd;
    std::mt19937 gen(42); // 固定种子
    std::uniform_real_distribution<double> dis(-1.0, 1.0);

    for (auto &val : mat)
    {
        val = dis(gen);
    }
}

// MLP前向传播
void mlp_forward(const std::vector<double> &X, const std::vector<double> &W1, const std::vector<double> &B1,
                 const std::vector<double> &W2, const std::vector<double> &B2,
                 std::vector<double> &Y)
{
    std::vector<double> H(BATCH * H_SIZE);

    // 第一层：H = X * W1 + B1
    matmul_cpu(X, W1, H, BATCH, H_SIZE, I);
    add_bias_cpu(H, B1, BATCH, H_SIZE);
    relu_cpu(H, BATCH * H_SIZE);

    // 第二层：Y = H * W2 + B2
    matmul_cpu(H, W2, Y, BATCH, O, H_SIZE);
    add_bias_cpu(Y, B2, BATCH, O);
}

// 优化版本的MLP前向传播（使用分块）
void mlp_forward_optimized(const std::vector<double> &X, const std::vector<double> &W1, const std::vector<double> &B1,
                           const std::vector<double> &W2, const std::vector<double> &B2,
                           std::vector<double> &Y)
{
    std::vector<double> H(BATCH * H_SIZE);
    const int BLOCK_SIZE = 32;

    // 第一层：H = X * W1 + B1 (分块优化)
    for (int i = 0; i < BATCH; i += BLOCK_SIZE)
    {
        for (int j = 0; j < H_SIZE; j += BLOCK_SIZE)
        {
            for (int k = 0; k < I; k += BLOCK_SIZE)
            {
                for (int ii = i; ii < std::min(i + BLOCK_SIZE, BATCH); ++ii)
                {
                    for (int jj = j; jj < std::min(j + BLOCK_SIZE, H_SIZE); ++jj)
                    {
                        if (k == 0)
                            H[ii * H_SIZE + jj] = 0.0;
                        for (int kk = k; kk < std::min(k + BLOCK_SIZE, I); ++kk)
                        {
                            H[ii * H_SIZE + jj] += X[ii * I + kk] * W1[kk * H_SIZE + jj];
                        }
                    }
                }
            }
        }
    }

    // 添加偏置和ReLU
    add_bias_cpu(H, B1, BATCH, H_SIZE);
    relu_cpu(H, BATCH * H_SIZE);

    // 第二层：Y = H * W2 + B2 (分块优化)
    for (int i = 0; i < BATCH; i += BLOCK_SIZE)
    {
        for (int j = 0; j < O; j += BLOCK_SIZE)
        {
            for (int k = 0; k < H_SIZE; k += BLOCK_SIZE)
            {
                for (int ii = i; ii < std::min(i + BLOCK_SIZE, BATCH); ++ii)
                {
                    for (int jj = j; jj < std::min(j + BLOCK_SIZE, O); ++jj)
                    {
                        if (k == 0)
                            Y[ii * O + jj] = 0.0;
                        for (int kk = k; kk < std::min(k + BLOCK_SIZE, H_SIZE); ++kk)
                        {
                            Y[ii * O + jj] += H[ii * H_SIZE + kk] * W2[kk * O + jj];
                        }
                    }
                }
            }
        }
    }

    add_bias_cpu(Y, B2, BATCH, O);
}

// 验证结果
bool validate_results(const std::vector<double> &result1, const std::vector<double> &result2, double tol = 1e-6)
{
    for (size_t i = 0; i < result1.size(); ++i)
    {
        if (std::abs(result1[i] - result2[i]) > tol)
        {
            std::cout << "Validation failed at index " << i
                      << ": result1=" << result1[i]
                      << ", result2=" << result2[i]
                      << ", diff=" << std::abs(result1[i] - result2[i]) << std::endl;
            return false;
        }
    }
    return true;
}

int main()
{
    // 初始化数据
    std::vector<double> X(BATCH * I), W1(I * H_SIZE), B1(H_SIZE), W2(H_SIZE * O), B2(O);
    std::vector<double> Y_basic(BATCH * O), Y_optimized(BATCH * O);

    random_init(X);
    random_init(W1);
    random_init(B1);
    random_init(W2);
    random_init(B2);

    std::cout << "MLP Forward Propagation - CPU Implementation Demo" << std::endl;
    std::cout << "Network: " << BATCH << "×" << I << " → " << I << "×" << H_SIZE << " (ReLU) → " << H_SIZE << "×" << O << std::endl;

    // 基础版本性能测试
    auto start1 = std::chrono::high_resolution_clock::now();
    mlp_forward(X, W1, B1, W2, B2, Y_basic);
    auto end1 = std::chrono::high_resolution_clock::now();
    auto time1 = std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - start1).count() / 1000000.0;

    // 优化版本性能测试
    auto start2 = std::chrono::high_resolution_clock::now();
    mlp_forward_optimized(X, W1, B1, W2, B2, Y_optimized);
    auto end2 = std::chrono::high_resolution_clock::now();
    auto time2 = std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count() / 1000000.0;

    // 性能统计
    std::cout << "\n=== CPU Performance Results ===" << std::endl;
    std::cout << "Basic CPU Time: " << time1 << " ms" << std::endl;
    std::cout << "Optimized CPU Time: " << time2 << " ms" << std::endl;
    std::cout << "CPU Optimization Speedup: " << (double)time1 / time2 << "x" << std::endl;

    // 验证结果
    if (validate_results(Y_basic, Y_optimized))
    {
        std::cout << "\n✓ Validation PASSED: Optimized results match basic implementation" << std::endl;
    }
    else
    {
        std::cout << "\n✗ Validation FAILED: Results differ between implementations" << std::endl;
    }

    // 模拟DCU性能（基于理论计算）
    double theoretical_dcu_time = time1 / 100.0; // 假设DCU快100倍
    std::cout << "\n=== Simulated DCU Performance ===" << std::endl;
    std::cout << "Theoretical DCU Time: " << theoretical_dcu_time << " ms" << std::endl;
    std::cout << "Theoretical DCU Speedup: " << (double)time1 / theoretical_dcu_time << "x" << std::endl;

    // 打印部分输出结果
    std::cout << "\n=== Sample Outputs ===" << std::endl;
    for (int i = 0; i < 3; ++i)
    {
        std::cout << "Batch[" << i << "]: ";
        for (int j = 0; j < O; ++j)
        {
            std::cout << Y_basic[i * O + j] << " ";
        }
        std::cout << std::endl;
    }

    // 输出性能数据用于绘图
    std::cout << "\n=== Performance Data for Plotting ===" << std::endl;
    std::cout << "CPU_Basic," << time1 << std::endl;
    std::cout << "CPU_Optimized," << time2 << std::endl;
    std::cout << "DCU_Theoretical," << theoretical_dcu_time << std::endl;

    return 0;
}