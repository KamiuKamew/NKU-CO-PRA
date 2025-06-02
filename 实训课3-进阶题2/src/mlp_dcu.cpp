#include <hip/hip_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <string>
#include <sstream>
#include <algorithm>
#include <random>

// 网络架构参数
#define INPUT_DIM 10        // 滑动窗口大小
#define HIDDEN_DIM 64       // 隐藏层神经元数量
#define OUTPUT_DIM 1        // 输出维度
#define BATCH_SIZE 128      // 批处理大小
#define EPOCHS 500          // 训练轮数
#define LEARNING_RATE 0.001 // 学习率
#define TRAIN_RATIO 0.8     // 训练集比例

// HIP错误检查宏
#define HIP_CHECK(call)                                                                                                  \
    do                                                                                                                   \
    {                                                                                                                    \
        hipError_t err = call;                                                                                           \
        if (err != hipSuccess)                                                                                           \
        {                                                                                                                \
            std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ << " - " << hipGetErrorString(err) << std::endl; \
            exit(1);                                                                                                     \
        }                                                                                                                \
    } while (0)

// ================================ HIP Kernels ================================

// 矩阵乘法 kernel: C = A * B
__global__ void matmul_kernel(const double *A, const double *B, double *C, int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N)
    {
        double sum = 0.0;
        for (int k = 0; k < K; k++)
        {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// 矩阵加法 kernel: C = A + B (广播偏置)
__global__ void add_bias_kernel(double *output, const double *bias, int rows, int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;

    if (idx < total)
    {
        int col = idx % cols;
        output[idx] += bias[col];
    }
}

// ReLU激活函数 kernel
__global__ void relu_kernel(double *data, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        data[idx] = fmax(0.0, data[idx]);
    }
}

// ReLU反向传播 kernel
__global__ void relu_backward_kernel(double *grad, const double *input, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        grad[idx] = (input[idx] > 0.0) ? grad[idx] : 0.0;
    }
}

// MSE损失计算 kernel
__global__ void mse_loss_kernel(const double *pred, const double *target, double *loss, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        double diff = pred[idx] - target[idx];
        loss[idx] = diff * diff;
    }
}

// 输出层梯度计算 kernel (MSE)
__global__ void output_grad_kernel(const double *pred, const double *target, double *grad, int batch_size, int output_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * output_dim;
    if (idx < total_size)
    {
        grad[idx] = 2.0 * (pred[idx] - target[idx]) / batch_size;
    }
}

// 偏置梯度计算 kernel (对batch维度求和)
__global__ void reduce_bias_grad_kernel(const double *grad_output, double *grad_bias, int batch_size, int output_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < output_dim)
    {
        double sum = 0.0;
        for (int i = 0; i < batch_size; i++)
        {
            sum += grad_output[i * output_dim + idx];
        }
        grad_bias[idx] = sum / batch_size; // 归一化
    }
}

// 梯度缩放 kernel
__global__ void scale_gradient_kernel(double *grad, double scale, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        grad[idx] *= scale;
    }
}

// 梯度裁剪 kernel
__global__ void clip_gradient_kernel(double *grad, int size, double clip_value)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        if (grad[idx] > clip_value)
            grad[idx] = clip_value;
        else if (grad[idx] < -clip_value)
            grad[idx] = -clip_value;
    }
}

// SGD权重更新 kernel
__global__ void sgd_update_kernel(double *weights, const double *grad, double lr, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        weights[idx] -= lr * grad[idx];
    }
}

// 矩阵转置 kernel
__global__ void transpose_kernel(const double *input, double *output, int rows, int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols)
    {
        int row = idx / cols;
        int col = idx % cols;
        output[col * rows + row] = input[row * cols + col];
    }
}

// ================================ MLP 类定义 ================================

class MLPNetwork
{
private:
    // 主机端参数
    double *h_W1, *h_b1, *h_W2, *h_b2;

    // 设备端参数
    double *d_W1, *d_b1, *d_W2, *d_b2;

    // 设备端前向传播缓存
    double *d_hidden, *d_output;

    // 设备端反向传播梯度
    double *d_grad_W1, *d_grad_b1, *d_grad_W2, *d_grad_b2;
    double *d_grad_hidden, *d_grad_output;

    // 设备端临时缓存
    double *d_W1_T, *d_W2_T, *d_hidden_no_relu;

public:
    MLPNetwork()
    {
        // 分配主机内存
        h_W1 = new double[INPUT_DIM * HIDDEN_DIM];
        h_b1 = new double[HIDDEN_DIM];
        h_W2 = new double[HIDDEN_DIM * OUTPUT_DIM];
        h_b2 = new double[OUTPUT_DIM];

        // 随机初始化权重 (Xavier初始化)
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> dis_W1(0.0, sqrt(2.0 / INPUT_DIM));
        std::normal_distribution<double> dis_W2(0.0, sqrt(2.0 / HIDDEN_DIM));

        for (int i = 0; i < INPUT_DIM * HIDDEN_DIM; i++)
            h_W1[i] = dis_W1(gen);
        for (int i = 0; i < HIDDEN_DIM; i++)
            h_b1[i] = 0.0;
        for (int i = 0; i < HIDDEN_DIM * OUTPUT_DIM; i++)
            h_W2[i] = dis_W2(gen);
        for (int i = 0; i < OUTPUT_DIM; i++)
            h_b2[i] = 0.0;

        // 分配设备内存
        HIP_CHECK(hipMalloc(&d_W1, INPUT_DIM * HIDDEN_DIM * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_b1, HIDDEN_DIM * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_W2, HIDDEN_DIM * OUTPUT_DIM * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_b2, OUTPUT_DIM * sizeof(double)));

        HIP_CHECK(hipMalloc(&d_hidden, BATCH_SIZE * HIDDEN_DIM * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_output, BATCH_SIZE * OUTPUT_DIM * sizeof(double)));

        HIP_CHECK(hipMalloc(&d_grad_W1, INPUT_DIM * HIDDEN_DIM * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_grad_b1, HIDDEN_DIM * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_grad_W2, HIDDEN_DIM * OUTPUT_DIM * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_grad_b2, OUTPUT_DIM * sizeof(double)));

        HIP_CHECK(hipMalloc(&d_grad_hidden, BATCH_SIZE * HIDDEN_DIM * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_grad_output, BATCH_SIZE * OUTPUT_DIM * sizeof(double)));

        HIP_CHECK(hipMalloc(&d_W1_T, HIDDEN_DIM * INPUT_DIM * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_W2_T, OUTPUT_DIM * HIDDEN_DIM * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_hidden_no_relu, BATCH_SIZE * HIDDEN_DIM * sizeof(double)));

        // 拷贝初始权重到设备
        copyWeightsToDevice();
    }

    ~MLPNetwork()
    {
        // 释放主机内存
        delete[] h_W1;
        delete[] h_b1;
        delete[] h_W2;
        delete[] h_b2;

        // 释放设备内存
        hipFree(d_W1);
        hipFree(d_b1);
        hipFree(d_W2);
        hipFree(d_b2);
        hipFree(d_hidden);
        hipFree(d_output);
        hipFree(d_grad_W1);
        hipFree(d_grad_b1);
        hipFree(d_grad_W2);
        hipFree(d_grad_b2);
        hipFree(d_grad_hidden);
        hipFree(d_grad_output);
        hipFree(d_W1_T);
        hipFree(d_W2_T);
        hipFree(d_hidden_no_relu);
    }

    void copyWeightsToDevice()
    {
        HIP_CHECK(hipMemcpy(d_W1, h_W1, INPUT_DIM * HIDDEN_DIM * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_b1, h_b1, HIDDEN_DIM * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_W2, h_W2, HIDDEN_DIM * OUTPUT_DIM * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_b2, h_b2, OUTPUT_DIM * sizeof(double), hipMemcpyHostToDevice));
    }

    void copyWeightsToHost()
    {
        HIP_CHECK(hipMemcpy(h_W1, d_W1, INPUT_DIM * HIDDEN_DIM * sizeof(double), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_b1, d_b1, HIDDEN_DIM * sizeof(double), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_W2, d_W2, HIDDEN_DIM * OUTPUT_DIM * sizeof(double), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_b2, d_b2, OUTPUT_DIM * sizeof(double), hipMemcpyDeviceToHost));
    }

    // 前向传播
    void forward(const double *d_input)
    {
        dim3 block(16, 16);
        dim3 grid_hidden((HIDDEN_DIM + block.x - 1) / block.x, (BATCH_SIZE + block.y - 1) / block.y);
        dim3 grid_output((OUTPUT_DIM + block.x - 1) / block.x, (BATCH_SIZE + block.y - 1) / block.y);

        // 第一层: hidden = input * W1 + b1
        matmul_kernel<<<grid_hidden, block>>>(d_input, d_W1, d_hidden_no_relu, BATCH_SIZE, HIDDEN_DIM, INPUT_DIM);
        HIP_CHECK(hipDeviceSynchronize());

        int bias_threads = (BATCH_SIZE * HIDDEN_DIM + 255) / 256;
        add_bias_kernel<<<bias_threads, 256>>>(d_hidden_no_relu, d_b1, BATCH_SIZE, HIDDEN_DIM);
        HIP_CHECK(hipDeviceSynchronize());

        // 复制到hidden并应用ReLU
        HIP_CHECK(hipMemcpy(d_hidden, d_hidden_no_relu, BATCH_SIZE * HIDDEN_DIM * sizeof(double), hipMemcpyDeviceToDevice));
        int relu_threads = (BATCH_SIZE * HIDDEN_DIM + 255) / 256;
        relu_kernel<<<relu_threads, 256>>>(d_hidden, BATCH_SIZE * HIDDEN_DIM);
        HIP_CHECK(hipDeviceSynchronize());

        // 第二层: output = hidden * W2 + b2
        matmul_kernel<<<grid_output, block>>>(d_hidden, d_W2, d_output, BATCH_SIZE, OUTPUT_DIM, HIDDEN_DIM);
        HIP_CHECK(hipDeviceSynchronize());

        bias_threads = (BATCH_SIZE * OUTPUT_DIM + 255) / 256;
        add_bias_kernel<<<bias_threads, 256>>>(d_output, d_b2, BATCH_SIZE, OUTPUT_DIM);
        HIP_CHECK(hipDeviceSynchronize());
    }

    // 反向传播
    void backward(const double *d_input, const double *d_target)
    {
        dim3 block(16, 16);

        // 计算输出层梯度 (已经归一化)
        int threads = (BATCH_SIZE * OUTPUT_DIM + 255) / 256;
        output_grad_kernel<<<threads, 256>>>(d_output, d_target, d_grad_output, BATCH_SIZE, OUTPUT_DIM);
        HIP_CHECK(hipDeviceSynchronize());

        // 计算W2梯度: grad_W2 = hidden^T * grad_output
        dim3 grid_T((INPUT_DIM + block.x - 1) / block.x, (HIDDEN_DIM + block.y - 1) / block.y);
        transpose_kernel<<<grid_T, block>>>(d_hidden, d_W2_T, BATCH_SIZE, HIDDEN_DIM);
        HIP_CHECK(hipDeviceSynchronize());

        dim3 grid_grad_W2((OUTPUT_DIM + block.x - 1) / block.x, (HIDDEN_DIM + block.y - 1) / block.y);
        matmul_kernel<<<grid_grad_W2, block>>>(d_W2_T, d_grad_output, d_grad_W2, HIDDEN_DIM, OUTPUT_DIM, BATCH_SIZE);
        HIP_CHECK(hipDeviceSynchronize());

        // 归一化W2梯度
        double scale = 1.0 / BATCH_SIZE;
        int w2_threads = (HIDDEN_DIM * OUTPUT_DIM + 255) / 256;
        scale_gradient_kernel<<<w2_threads, 256>>>(d_grad_W2, scale, HIDDEN_DIM * OUTPUT_DIM);
        HIP_CHECK(hipDeviceSynchronize());

        // 计算b2梯度 (对batch维度求和并归一化)
        int b2_threads = (OUTPUT_DIM + 255) / 256;
        reduce_bias_grad_kernel<<<b2_threads, 256>>>(d_grad_output, d_grad_b2, BATCH_SIZE, OUTPUT_DIM);
        HIP_CHECK(hipDeviceSynchronize());

        // 计算隐藏层梯度: grad_hidden = grad_output * W2^T
        transpose_kernel<<<grid_grad_W2, block>>>(d_W2, d_W2_T, HIDDEN_DIM, OUTPUT_DIM);
        HIP_CHECK(hipDeviceSynchronize());

        dim3 grid_grad_hidden((HIDDEN_DIM + block.x - 1) / block.x, (BATCH_SIZE + block.y - 1) / block.y);
        matmul_kernel<<<grid_grad_hidden, block>>>(d_grad_output, d_W2_T, d_grad_hidden, BATCH_SIZE, HIDDEN_DIM, OUTPUT_DIM);
        HIP_CHECK(hipDeviceSynchronize());

        // ReLU反向传播
        threads = (BATCH_SIZE * HIDDEN_DIM + 255) / 256;
        relu_backward_kernel<<<threads, 256>>>(d_grad_hidden, d_hidden_no_relu, BATCH_SIZE * HIDDEN_DIM);
        HIP_CHECK(hipDeviceSynchronize());

        // 计算W1梯度: grad_W1 = input^T * grad_hidden
        dim3 grid_T_input((HIDDEN_DIM + block.x - 1) / block.x, (INPUT_DIM + block.y - 1) / block.y);
        transpose_kernel<<<grid_T_input, block>>>(d_input, d_W1_T, BATCH_SIZE, INPUT_DIM);
        HIP_CHECK(hipDeviceSynchronize());

        dim3 grid_grad_W1((HIDDEN_DIM + block.x - 1) / block.x, (INPUT_DIM + block.y - 1) / block.y);
        matmul_kernel<<<grid_grad_W1, block>>>(d_W1_T, d_grad_hidden, d_grad_W1, INPUT_DIM, HIDDEN_DIM, BATCH_SIZE);
        HIP_CHECK(hipDeviceSynchronize());

        // 归一化W1梯度
        int w1_threads = (INPUT_DIM * HIDDEN_DIM + 255) / 256;
        scale_gradient_kernel<<<w1_threads, 256>>>(d_grad_W1, scale, INPUT_DIM * HIDDEN_DIM);
        HIP_CHECK(hipDeviceSynchronize());

        // 计算b1梯度 (对batch维度求和并归一化)
        int b1_threads = (HIDDEN_DIM + 255) / 256;
        reduce_bias_grad_kernel<<<b1_threads, 256>>>(d_grad_hidden, d_grad_b1, BATCH_SIZE, HIDDEN_DIM);
        HIP_CHECK(hipDeviceSynchronize());

        // 梯度裁剪 (防止梯度爆炸)
        double clip_value = 5.0;
        clip_gradient_kernel<<<w1_threads, 256>>>(d_grad_W1, INPUT_DIM * HIDDEN_DIM, clip_value);
        clip_gradient_kernel<<<b1_threads, 256>>>(d_grad_b1, HIDDEN_DIM, clip_value);
        clip_gradient_kernel<<<w2_threads, 256>>>(d_grad_W2, HIDDEN_DIM * OUTPUT_DIM, clip_value);
        clip_gradient_kernel<<<b2_threads, 256>>>(d_grad_b2, OUTPUT_DIM, clip_value);
        HIP_CHECK(hipDeviceSynchronize());
    }

    // 更新参数
    void updateWeights(double lr)
    {
        int threads_W1 = (INPUT_DIM * HIDDEN_DIM + 255) / 256;
        int threads_b1 = (HIDDEN_DIM + 255) / 256;
        int threads_W2 = (HIDDEN_DIM * OUTPUT_DIM + 255) / 256;
        int threads_b2 = (OUTPUT_DIM + 255) / 256;

        sgd_update_kernel<<<threads_W1, 256>>>(d_W1, d_grad_W1, lr, INPUT_DIM * HIDDEN_DIM);
        sgd_update_kernel<<<threads_b1, 256>>>(d_b1, d_grad_b1, lr, HIDDEN_DIM);
        sgd_update_kernel<<<threads_W2, 256>>>(d_W2, d_grad_W2, lr, HIDDEN_DIM * OUTPUT_DIM);
        sgd_update_kernel<<<threads_b2, 256>>>(d_b2, d_grad_b2, lr, OUTPUT_DIM);
        HIP_CHECK(hipDeviceSynchronize());
    }

    // 计算损失
    double computeLoss(const double *d_target)
    {
        double *d_loss;
        HIP_CHECK(hipMalloc(&d_loss, BATCH_SIZE * OUTPUT_DIM * sizeof(double)));

        int threads = (BATCH_SIZE * OUTPUT_DIM + 255) / 256;
        mse_loss_kernel<<<threads, 256>>>(d_output, d_target, d_loss, BATCH_SIZE * OUTPUT_DIM);
        HIP_CHECK(hipDeviceSynchronize());

        double *h_loss = new double[BATCH_SIZE * OUTPUT_DIM];
        HIP_CHECK(hipMemcpy(h_loss, d_loss, BATCH_SIZE * OUTPUT_DIM * sizeof(double), hipMemcpyDeviceToHost));

        double total_loss = 0.0;
        for (int i = 0; i < BATCH_SIZE * OUTPUT_DIM; i++)
        {
            total_loss += h_loss[i];
        }

        delete[] h_loss;
        hipFree(d_loss);
        return total_loss / (BATCH_SIZE * OUTPUT_DIM);
    }

    // 获取输出
    void getOutput(double *output)
    {
        HIP_CHECK(hipMemcpy(output, d_output, BATCH_SIZE * OUTPUT_DIM * sizeof(double), hipMemcpyDeviceToHost));
    }

    // 推理专用前向传播 (使用独立输出缓冲区)
    void forward_infer(const double *d_input, double *d_out, int batch_size)
    {
        dim3 block(16, 16);
        dim3 grid_hidden((HIDDEN_DIM + block.x - 1) / block.x, (batch_size + block.y - 1) / block.y);
        dim3 grid_output((OUTPUT_DIM + block.x - 1) / block.x, (batch_size + block.y - 1) / block.y);

        // 分配临时隐藏层缓冲区（推理时不保存状态）
        double *d_temp_hidden, *d_temp_hidden_no_relu;
        HIP_CHECK(hipMalloc(&d_temp_hidden, batch_size * HIDDEN_DIM * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_temp_hidden_no_relu, batch_size * HIDDEN_DIM * sizeof(double)));

        // 第一层: hidden = input * W1 + b1
        matmul_kernel<<<grid_hidden, block>>>(d_input, d_W1, d_temp_hidden_no_relu, batch_size, HIDDEN_DIM, INPUT_DIM);
        HIP_CHECK(hipDeviceSynchronize());

        int bias_threads = (batch_size * HIDDEN_DIM + 255) / 256;
        add_bias_kernel<<<bias_threads, 256>>>(d_temp_hidden_no_relu, d_b1, batch_size, HIDDEN_DIM);
        HIP_CHECK(hipDeviceSynchronize());

        // 复制到hidden并应用ReLU
        HIP_CHECK(hipMemcpy(d_temp_hidden, d_temp_hidden_no_relu, batch_size * HIDDEN_DIM * sizeof(double), hipMemcpyDeviceToDevice));
        int relu_threads = (batch_size * HIDDEN_DIM + 255) / 256;
        relu_kernel<<<relu_threads, 256>>>(d_temp_hidden, batch_size * HIDDEN_DIM);
        HIP_CHECK(hipDeviceSynchronize());

        // 第二层: output = hidden * W2 + b2
        matmul_kernel<<<grid_output, block>>>(d_temp_hidden, d_W2, d_out, batch_size, OUTPUT_DIM, HIDDEN_DIM);
        HIP_CHECK(hipDeviceSynchronize());

        bias_threads = (batch_size * OUTPUT_DIM + 255) / 256;
        add_bias_kernel<<<bias_threads, 256>>>(d_out, d_b2, batch_size, OUTPUT_DIM);
        HIP_CHECK(hipDeviceSynchronize());

        // 清理临时缓冲区
        hipFree(d_temp_hidden);
        hipFree(d_temp_hidden_no_relu);
    }

    // 推理函数（修复版本）
    void predict(const double *d_input, double *output, int batch_size)
    {
        // 分配独立的输出缓冲区
        double *d_temp_output;
        HIP_CHECK(hipMalloc(&d_temp_output, batch_size * OUTPUT_DIM * sizeof(double)));

        // 使用独立的前向传播，不影响训练状态
        forward_infer(d_input, d_temp_output, batch_size);

        // 拷贝结果到主机
        HIP_CHECK(hipMemcpy(output, d_temp_output, batch_size * OUTPUT_DIM * sizeof(double), hipMemcpyDeviceToHost));

        // 清理
        hipFree(d_temp_output);
    }
};

// ================================ 数据处理函数 ================================

// 加载JSON带宽数据
std::vector<double> load_json_bandwidth(const std::string &filename)
{
    std::ifstream file(filename);
    std::vector<double> data;

    if (!file.is_open())
    {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return data;
    }

    std::string line;
    std::getline(file, line);

    // 移除首尾的方括号
    line = line.substr(1, line.length() - 2);

    std::stringstream ss(line);
    std::string token;

    while (std::getline(ss, token, ','))
    {
        // 移除空格
        token.erase(std::remove_if(token.begin(), token.end(), ::isspace), token.end());
        if (!token.empty())
        {
            data.push_back(std::stod(token));
        }
    }

    std::cout << "加载了 " << data.size() << " 个带宽数据点" << std::endl;
    return data;
}

// 数据归一化
void normalize_data(std::vector<double> &data, double &min_val, double &max_val)
{
    min_val = *std::min_element(data.begin(), data.end());
    max_val = *std::max_element(data.begin(), data.end());

    for (auto &val : data)
    {
        val = (val - min_val) / (max_val - min_val);
    }

    std::cout << "数据归一化完成, 范围: [" << min_val << ", " << max_val << "]" << std::endl;
}

// 数据反归一化
void denormalize_data(std::vector<double> &data, double min_val, double max_val)
{
    for (auto &val : data)
    {
        val = val * (max_val - min_val) + min_val;
    }
}

// 创建滑动窗口数据集
void create_dataset(const std::vector<double> &data, std::vector<double> &X, std::vector<double> &y)
{
    int num_samples = data.size() - INPUT_DIM;

    X.resize(num_samples * INPUT_DIM);
    y.resize(num_samples);

    for (int i = 0; i < num_samples; i++)
    {
        // 输入：连续INPUT_DIM个时间点
        for (int j = 0; j < INPUT_DIM; j++)
        {
            X[i * INPUT_DIM + j] = data[i + j];
        }
        // 目标：下一个时间点
        y[i] = data[i + INPUT_DIM];
    }

    std::cout << "创建了 " << num_samples << " 个训练样本" << std::endl;
}

// ================================ 主函数 ================================

int main()
{
    std::cout << "=== 基于DCU的MLP低轨卫星带宽预测系统 ===" << std::endl;

    // 1. 加载和预处理数据
    std::vector<double> raw_data = load_json_bandwidth("../data/starlink_bw.json");
    if (raw_data.empty())
    {
        std::cerr << "数据加载失败!" << std::endl;
        return -1;
    }

    double min_val, max_val;
    normalize_data(raw_data, min_val, max_val);

    // 2. 创建数据集
    std::vector<double> X, y;
    create_dataset(raw_data, X, y);

    int num_samples = y.size();
    int train_size = static_cast<int>(num_samples * TRAIN_RATIO);
    int test_size = num_samples - train_size;

    std::cout << "训练集大小: " << train_size << ", 测试集大小: " << test_size << std::endl;

    // 3. 分配设备内存
    double *d_X_train, *d_y_train, *d_X_test, *d_y_test;
    HIP_CHECK(hipMalloc(&d_X_train, train_size * INPUT_DIM * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_y_train, train_size * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_X_test, test_size * INPUT_DIM * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_y_test, test_size * sizeof(double)));

    // 拷贝数据到设备
    HIP_CHECK(hipMemcpy(d_X_train, X.data(), train_size * INPUT_DIM * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_y_train, y.data(), train_size * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_X_test, X.data() + train_size * INPUT_DIM, test_size * INPUT_DIM * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_y_test, y.data() + train_size, test_size * sizeof(double), hipMemcpyHostToDevice));

    // 4. 创建和训练模型
    MLPNetwork model;

    std::cout << "\n=== 开始训练 ===" << std::endl;
    auto train_start = std::chrono::high_resolution_clock::now();

    double best_loss = 1e10;
    int patience = 0;
    const int max_patience = 50; // early stopping patience

    for (int epoch = 0; epoch < EPOCHS; epoch++)
    {
        double total_loss = 0.0;
        int num_batches = (train_size + BATCH_SIZE - 1) / BATCH_SIZE;

        for (int batch = 0; batch < num_batches; batch++)
        {
            int start_idx = batch * BATCH_SIZE;
            int current_batch_size = std::min(BATCH_SIZE, train_size - start_idx);

            if (current_batch_size < BATCH_SIZE)
                break; // 跳过最后一个不完整的批次

            // 前向传播
            model.forward(d_X_train + start_idx * INPUT_DIM);

            // 计算损失
            double batch_loss = model.computeLoss(d_y_train + start_idx);

            // NaN检查
            if (std::isnan(batch_loss) || std::isinf(batch_loss))
            {
                std::cerr << "检测到NaN或Inf损失，提前停止训练! Epoch: " << epoch << ", Batch: " << batch << std::endl;
                goto training_complete;
            }

            // 损失爆炸检查
            if (batch_loss > 1e6)
            {
                std::cerr << "损失过大 (" << batch_loss << ")，提前终止! Epoch: " << epoch << ", Batch: " << batch << std::endl;
                goto training_complete;
            }

            total_loss += batch_loss;

            // 反向传播
            model.backward(d_X_train + start_idx * INPUT_DIM, d_y_train + start_idx);

            // 更新参数
            model.updateWeights(LEARNING_RATE);
        }

        double avg_loss = total_loss / num_batches;

        // Early stopping检查
        if (avg_loss < best_loss)
        {
            best_loss = avg_loss;
            patience = 0;
        }
        else
        {
            patience++;
            if (patience >= max_patience)
            {
                std::cout << "Early stopping triggered at epoch " << epoch + 1 << std::endl;
                break;
            }
        }

        if ((epoch + 1) % 10 == 0 || epoch < 20)
        {
            std::cout << "Epoch " << epoch + 1 << "/" << EPOCHS
                      << ", 平均损失: " << avg_loss
                      << ", 最佳损失: " << best_loss
                      << ", Patience: " << patience << std::endl;
        }
    }

training_complete:
    auto train_end = std::chrono::high_resolution_clock::now();
    double train_time = std::chrono::duration<double, std::milli>(train_end - train_start).count();

    std::cout << "训练完成! 训练时间: " << train_time << " ms, 最终最佳损失: " << best_loss << std::endl;

    // 5. 测试推理性能
    std::cout << "\n=== 开始推理测试 ===" << std::endl;

    auto infer_start = std::chrono::high_resolution_clock::now();

    // 预测前10个测试样本
    int test_samples = std::min(10, test_size);
    std::vector<double> predictions(test_samples);

    std::cout << "开始逐样本推理测试..." << std::endl;
    for (int i = 0; i < test_samples; i++)
    {
        double pred;
        model.predict(d_X_test + i * INPUT_DIM, &pred, 1);
        predictions[i] = pred;

        // 添加调试信息：打印归一化预测值
        std::cout << "样本 " << i + 1 << " 归一化预测值: " << pred << std::endl;
    }

    auto infer_end = std::chrono::high_resolution_clock::now();
    double infer_time = std::chrono::duration<double, std::milli>(infer_end - infer_start).count();

    // 6. 计算测试误差
    std::vector<double> h_y_test(test_samples);
    HIP_CHECK(hipMemcpy(h_y_test.data(), d_y_test, test_samples * sizeof(double), hipMemcpyDeviceToHost));

    double mse = 0.0;
    for (int i = 0; i < test_samples; i++)
    {
        double diff = predictions[i] - h_y_test[i];
        mse += diff * diff;
    }
    mse /= test_samples;

    // 反归一化显示结果
    std::vector<double> pred_denorm = predictions;
    std::vector<double> actual_denorm = h_y_test;
    denormalize_data(pred_denorm, min_val, max_val);
    denormalize_data(actual_denorm, min_val, max_val);

    // 7. 输出结果
    std::cout << "\n=== 性能评测结果 ===" << std::endl;
    std::cout << "训练时间: " << train_time << " ms" << std::endl;
    std::cout << "推理时间: " << infer_time << " ms (" << test_samples << " 个样本)" << std::endl;
    std::cout << "平均每样本推理时间: " << infer_time / test_samples << " ms" << std::endl;
    std::cout << "推理吞吐量: " << test_samples * 1000.0 / infer_time << " 样本/秒" << std::endl;
    std::cout << "归一化MSE: " << mse << std::endl;
    std::cout << "最终训练损失: " << best_loss << std::endl;

    std::cout << "\n=== 预测结果对比 (前10个样本) ===" << std::endl;
    for (int i = 0; i < test_samples; i++)
    {
        std::cout << "样本 " << i + 1 << " - 预测: " << pred_denorm[i]
                  << " Mbps, 实际: " << actual_denorm[i]
                  << " Mbps, 误差: " << abs(pred_denorm[i] - actual_denorm[i]) << " Mbps" << std::endl;
    }

    // 清理内存
    hipFree(d_X_train);
    hipFree(d_y_train);
    hipFree(d_X_test);
    hipFree(d_y_test);

    std::cout << "\n=== 系统运行完成 ===" << std::endl;
    return 0;
}