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

int main()
{
    // 初始化数据
    std::vector<double> X(BATCH * I), W1(I * H_SIZE), B1(H_SIZE), W2(H_SIZE * O), B2(O);
    std::vector<double> Y_basic(BATCH * O);

    random_init(X);
    random_init(W1);
    random_init(B1);
    random_init(W2);
    random_init(B2);

    std::cout << "=== CPU Debug Version ===" << std::endl;
    std::cout << "Network: " << BATCH << "×" << I << " → " << I << "×" << H_SIZE << " (ReLU) → " << H_SIZE << "×" << O << std::endl;

    // 多次测试不同的时间精度
    std::cout << "\n=== 时间测量测试 ===" << std::endl;

    // 1. 测试纳秒精度
    auto start_ns = std::chrono::high_resolution_clock::now();
    mlp_forward(X, W1, B1, W2, B2, Y_basic);
    auto end_ns = std::chrono::high_resolution_clock::now();
    auto time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_ns - start_ns).count();

    // 2. 测试微秒精度
    auto start_us = std::chrono::high_resolution_clock::now();
    mlp_forward(X, W1, B1, W2, B2, Y_basic);
    auto end_us = std::chrono::high_resolution_clock::now();
    auto time_us = std::chrono::duration_cast<std::chrono::microseconds>(end_us - start_us).count();

    // 3. 测试毫秒精度
    auto start_ms = std::chrono::high_resolution_clock::now();
    mlp_forward(X, W1, B1, W2, B2, Y_basic);
    auto end_ms = std::chrono::high_resolution_clock::now();
    auto time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_ms - start_ms).count();

    std::cout << "纳秒测量: " << time_ns << " ns" << std::endl;
    std::cout << "微秒测量: " << time_us << " μs" << std::endl;
    std::cout << "毫秒测量: " << time_ms << " ms" << std::endl;
    std::cout << "转换为毫秒: " << (double)time_ns / 1000000.0 << " ms (from ns)" << std::endl;
    std::cout << "转换为毫秒: " << (double)time_us / 1000.0 << " ms (from μs)" << std::endl;

    // 4. 多次运行平均测试
    std::cout << "\n=== 多次运行测试 ===" << std::endl;
    double total_time = 0.0;
    int num_runs = 10;

    for (int i = 0; i < num_runs; ++i)
    {
        auto start = std::chrono::high_resolution_clock::now();
        mlp_forward(X, W1, B1, W2, B2, Y_basic);
        auto end = std::chrono::high_resolution_clock::now();
        double single_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000000.0;
        total_time += single_time;
        std::cout << "第" << (i + 1) << "次: " << single_time << " ms" << std::endl;
    }

    double avg_time = total_time / num_runs;
    std::cout << "平均时间: " << avg_time << " ms" << std::endl;

    // 5. 输出兼容原测试脚本的格式
    std::cout << "\n=== 标准输出格式 ===" << std::endl;
    std::cout << "Basic CPU Time: " << avg_time << " ms" << std::endl;

    return 0;
}