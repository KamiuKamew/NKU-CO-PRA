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
#define INPUT_DIM 10
#define HIDDEN_DIM 64
#define OUTPUT_DIM 1
#define BATCH_SIZE 128
#define EPOCHS 10000
#define LEARNING_RATE 0.0005
#define TRAIN_RATIO 0.8

// ================================ CPU MLP 实现 ================================

class CPUMLPNetwork
{
private:
    std::vector<double> W1, b1, W2, b2;
    std::vector<double> hidden, output;
    std::vector<double> grad_W1, grad_b1, grad_W2, grad_b2;
    std::vector<double> grad_hidden, grad_output;
    std::vector<double> hidden_no_relu;

public:
    CPUMLPNetwork()
    {
        // 初始化权重和偏置
        W1.resize(INPUT_DIM * HIDDEN_DIM);
        b1.resize(HIDDEN_DIM);
        W2.resize(HIDDEN_DIM * OUTPUT_DIM);
        b2.resize(OUTPUT_DIM);

        // 分配前向传播缓存
        hidden.resize(BATCH_SIZE * HIDDEN_DIM);
        output.resize(BATCH_SIZE * OUTPUT_DIM);
        hidden_no_relu.resize(BATCH_SIZE * HIDDEN_DIM);

        // 分配梯度缓存
        grad_W1.resize(INPUT_DIM * HIDDEN_DIM);
        grad_b1.resize(HIDDEN_DIM);
        grad_W2.resize(HIDDEN_DIM * OUTPUT_DIM);
        grad_b2.resize(OUTPUT_DIM);
        grad_hidden.resize(BATCH_SIZE * HIDDEN_DIM);
        grad_output.resize(BATCH_SIZE * OUTPUT_DIM);

        // Xavier初始化
        std::random_device rd;
        std::mt19937 gen(42); // 使用固定种子确保一致性
        std::normal_distribution<double> dis_W1(0.0, sqrt(2.0 / INPUT_DIM));
        std::normal_distribution<double> dis_W2(0.0, sqrt(2.0 / HIDDEN_DIM));

        for (int i = 0; i < INPUT_DIM * HIDDEN_DIM; i++)
            W1[i] = dis_W1(gen);
        for (int i = 0; i < HIDDEN_DIM; i++)
            b1[i] = 0.0;
        for (int i = 0; i < HIDDEN_DIM * OUTPUT_DIM; i++)
            W2[i] = dis_W2(gen);
        for (int i = 0; i < OUTPUT_DIM; i++)
            b2[i] = 0.0;
    }

    // 矩阵乘法: C = A * B
    void matmul(const std::vector<double> &A, const std::vector<double> &B,
                std::vector<double> &C, int M, int N, int K)
    {
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                double sum = 0.0;
                for (int k = 0; k < K; k++)
                {
                    sum += A[i * K + k] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
    }

    // 前向传播
    void forward(const std::vector<double> &input)
    {
        // 第一层: hidden = input * W1 + b1
        matmul(input, W1, hidden_no_relu, BATCH_SIZE, HIDDEN_DIM, INPUT_DIM);

        for (int i = 0; i < BATCH_SIZE; i++)
        {
            for (int j = 0; j < HIDDEN_DIM; j++)
            {
                hidden_no_relu[i * HIDDEN_DIM + j] += b1[j];
            }
        }

        // ReLU激活
        for (int i = 0; i < BATCH_SIZE * HIDDEN_DIM; i++)
        {
            hidden[i] = std::max(0.0, hidden_no_relu[i]);
        }

        // 第二层: output = hidden * W2 + b2
        matmul(hidden, W2, output, BATCH_SIZE, OUTPUT_DIM, HIDDEN_DIM);

        for (int i = 0; i < BATCH_SIZE; i++)
        {
            for (int j = 0; j < OUTPUT_DIM; j++)
            {
                output[i * OUTPUT_DIM + j] += b2[j];
            }
        }
    }

    // 反向传播
    void backward(const std::vector<double> &input, const std::vector<double> &target)
    {
        // 计算输出层梯度 (已归一化)
        for (int i = 0; i < BATCH_SIZE * OUTPUT_DIM; i++)
        {
            grad_output[i] = 2.0 * (output[i] - target[i]) / BATCH_SIZE;
        }

        // 计算W2梯度
        std::fill(grad_W2.begin(), grad_W2.end(), 0.0);
        for (int i = 0; i < HIDDEN_DIM; i++)
        {
            for (int j = 0; j < OUTPUT_DIM; j++)
            {
                for (int b = 0; b < BATCH_SIZE; b++)
                {
                    grad_W2[i * OUTPUT_DIM + j] += hidden[b * HIDDEN_DIM + i] * grad_output[b * OUTPUT_DIM + j];
                }
                grad_W2[i * OUTPUT_DIM + j] /= BATCH_SIZE; // 归一化
            }
        }

        // 计算b2梯度 (对batch维度求和并归一化)
        std::fill(grad_b2.begin(), grad_b2.end(), 0.0);
        for (int j = 0; j < OUTPUT_DIM; j++)
        {
            for (int b = 0; b < BATCH_SIZE; b++)
            {
                grad_b2[j] += grad_output[b * OUTPUT_DIM + j];
            }
            grad_b2[j] /= BATCH_SIZE; // 归一化
        }

        // 计算隐藏层梯度
        std::fill(grad_hidden.begin(), grad_hidden.end(), 0.0);
        for (int i = 0; i < BATCH_SIZE; i++)
        {
            for (int j = 0; j < HIDDEN_DIM; j++)
            {
                for (int k = 0; k < OUTPUT_DIM; k++)
                {
                    grad_hidden[i * HIDDEN_DIM + j] += grad_output[i * OUTPUT_DIM + k] * W2[j * OUTPUT_DIM + k];
                }
            }
        }

        // ReLU反向传播
        for (int i = 0; i < BATCH_SIZE * HIDDEN_DIM; i++)
        {
            if (hidden_no_relu[i] <= 0.0)
            {
                grad_hidden[i] = 0.0;
            }
        }

        // 计算W1梯度
        std::fill(grad_W1.begin(), grad_W1.end(), 0.0);
        for (int i = 0; i < INPUT_DIM; i++)
        {
            for (int j = 0; j < HIDDEN_DIM; j++)
            {
                for (int b = 0; b < BATCH_SIZE; b++)
                {
                    grad_W1[i * HIDDEN_DIM + j] += input[b * INPUT_DIM + i] * grad_hidden[b * HIDDEN_DIM + j];
                }
                grad_W1[i * HIDDEN_DIM + j] /= BATCH_SIZE; // 归一化
            }
        }

        // 计算b1梯度 (对batch维度求和并归一化)
        std::fill(grad_b1.begin(), grad_b1.end(), 0.0);
        for (int j = 0; j < HIDDEN_DIM; j++)
        {
            for (int b = 0; b < BATCH_SIZE; b++)
            {
                grad_b1[j] += grad_hidden[b * HIDDEN_DIM + j];
            }
            grad_b1[j] /= BATCH_SIZE; // 归一化
        }

        // 梯度裁剪 (防止梯度爆炸)
        double clip_value = 5.0;
        for (int i = 0; i < INPUT_DIM * HIDDEN_DIM; i++)
        {
            if (grad_W1[i] > clip_value)
                grad_W1[i] = clip_value;
            else if (grad_W1[i] < -clip_value)
                grad_W1[i] = -clip_value;
        }
        for (int i = 0; i < HIDDEN_DIM; i++)
        {
            if (grad_b1[i] > clip_value)
                grad_b1[i] = clip_value;
            else if (grad_b1[i] < -clip_value)
                grad_b1[i] = -clip_value;
        }
        for (int i = 0; i < HIDDEN_DIM * OUTPUT_DIM; i++)
        {
            if (grad_W2[i] > clip_value)
                grad_W2[i] = clip_value;
            else if (grad_W2[i] < -clip_value)
                grad_W2[i] = -clip_value;
        }
        for (int i = 0; i < OUTPUT_DIM; i++)
        {
            if (grad_b2[i] > clip_value)
                grad_b2[i] = clip_value;
            else if (grad_b2[i] < -clip_value)
                grad_b2[i] = -clip_value;
        }
    }

    // 更新参数
    void updateWeights(double lr)
    {
        for (int i = 0; i < INPUT_DIM * HIDDEN_DIM; i++)
        {
            W1[i] -= lr * grad_W1[i];
        }
        for (int i = 0; i < HIDDEN_DIM; i++)
        {
            b1[i] -= lr * grad_b1[i];
        }
        for (int i = 0; i < HIDDEN_DIM * OUTPUT_DIM; i++)
        {
            W2[i] -= lr * grad_W2[i];
        }
        for (int i = 0; i < OUTPUT_DIM; i++)
        {
            b2[i] -= lr * grad_b2[i];
        }
    }

    // 计算损失
    double computeLoss(const std::vector<double> &target)
    {
        double loss = 0.0;
        for (int i = 0; i < BATCH_SIZE * OUTPUT_DIM; i++)
        {
            double diff = output[i] - target[i];
            loss += diff * diff;
        }
        return loss / (BATCH_SIZE * OUTPUT_DIM);
    }

    // 获取输出
    const std::vector<double> &getOutput() const
    {
        return output;
    }

    // 推理
    void predict(const std::vector<double> &input, std::vector<double> &pred, int batch_size)
    {
        // 创建独立的输入缓冲区（填充到BATCH_SIZE）
        std::vector<double> temp_input(BATCH_SIZE * INPUT_DIM, 0.0);
        std::vector<double> temp_output(BATCH_SIZE * OUTPUT_DIM, 0.0);
        std::vector<double> temp_hidden(BATCH_SIZE * HIDDEN_DIM, 0.0);
        std::vector<double> temp_hidden_no_relu(BATCH_SIZE * HIDDEN_DIM, 0.0);

        // 复制实际输入数据
        for (int i = 0; i < batch_size * INPUT_DIM; i++)
        {
            temp_input[i] = input[i];
        }

        // 独立的前向传播计算（不影响训练状态）
        // 第一层: hidden = input * W1 + b1
        matmul(temp_input, W1, temp_hidden_no_relu, BATCH_SIZE, HIDDEN_DIM, INPUT_DIM);

        for (int i = 0; i < BATCH_SIZE; i++)
        {
            for (int j = 0; j < HIDDEN_DIM; j++)
            {
                temp_hidden_no_relu[i * HIDDEN_DIM + j] += b1[j];
            }
        }

        // ReLU激活
        for (int i = 0; i < BATCH_SIZE * HIDDEN_DIM; i++)
        {
            temp_hidden[i] = std::max(0.0, temp_hidden_no_relu[i]);
        }

        // 第二层: output = hidden * W2 + b2
        matmul(temp_hidden, W2, temp_output, BATCH_SIZE, OUTPUT_DIM, HIDDEN_DIM);

        for (int i = 0; i < BATCH_SIZE; i++)
        {
            for (int j = 0; j < OUTPUT_DIM; j++)
            {
                temp_output[i * OUTPUT_DIM + j] += b2[j];
            }
        }

        // 复制结果
        for (int i = 0; i < batch_size; i++)
        {
            pred[i] = temp_output[i];
        }
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
    std::cout << "=== CPU基准版MLP低轨卫星带宽预测系统 ===" << std::endl;

    // 1. 加载和预处理数据
    std::vector<double> raw_data = load_json_bandwidth("data/starlink_bw.json");
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

    // 3. 创建和训练模型
    CPUMLPNetwork model;

    std::cout << "\n=== 开始CPU训练 ===" << std::endl;
    auto train_start = std::chrono::high_resolution_clock::now();

    double best_loss = 1e10;
    int patience = 0;
    const int max_patience = 20;         // 降低patience，更早停止
    const double min_improvement = 1e-6; // 最小改进阈值

    for (int epoch = 0; epoch < EPOCHS; epoch++)
    {
        double total_loss = 0.0;
        int num_batches = (train_size + BATCH_SIZE - 1) / BATCH_SIZE;

        for (int batch = 0; batch < num_batches; batch++)
        {
            int start_idx = batch * BATCH_SIZE;
            int current_batch_size = std::min(BATCH_SIZE, train_size - start_idx);

            if (current_batch_size < BATCH_SIZE)
                break;

            // 准备批次数据
            std::vector<double> batch_X(BATCH_SIZE * INPUT_DIM);
            std::vector<double> batch_y(BATCH_SIZE);

            for (int i = 0; i < BATCH_SIZE; i++)
            {
                for (int j = 0; j < INPUT_DIM; j++)
                {
                    batch_X[i * INPUT_DIM + j] = X[(start_idx + i) * INPUT_DIM + j];
                }
                batch_y[i] = y[start_idx + i];
            }

            // 前向传播
            model.forward(batch_X);

            // 计算损失
            double batch_loss = model.computeLoss(batch_y);

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
            model.backward(batch_X, batch_y);

            // 更新参数
            model.updateWeights(LEARNING_RATE);
        }

        double avg_loss = total_loss / num_batches;

        // Early stopping检查
        if (avg_loss < best_loss - min_improvement)
        {
            best_loss = avg_loss;
            patience = 0;
        }
        else
        {
            patience++;
            if (patience >= max_patience)
            {
                std::cout << "Early stopping triggered at epoch " << epoch + 1
                          << " (no improvement > " << min_improvement << " for "
                          << max_patience << " epochs)" << std::endl;
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

    std::cout << "CPU训练完成! 训练时间: " << train_time << " ms, 最终最佳损失: " << best_loss << std::endl;

    // 4. 测试推理性能
    std::cout << "\n=== 开始CPU推理测试 ===" << std::endl;

    auto infer_start = std::chrono::high_resolution_clock::now();

    // 预测前10个测试样本
    int test_samples = std::min(10, test_size);
    std::vector<double> predictions(test_samples);

    std::cout << "开始逐样本推理测试..." << std::endl;
    for (int i = 0; i < test_samples; i++)
    {
        std::vector<double> test_input(INPUT_DIM);
        for (int j = 0; j < INPUT_DIM; j++)
        {
            test_input[j] = X[(train_size + i) * INPUT_DIM + j];
        }

        std::vector<double> pred(1);
        model.predict(test_input, pred, 1);
        predictions[i] = pred[0];

        // 添加调试信息：打印归一化预测值
        std::cout << "样本 " << i + 1 << " 归一化预测值: " << pred[0] << std::endl;
    }

    auto infer_end = std::chrono::high_resolution_clock::now();
    double infer_time = std::chrono::duration<double, std::milli>(infer_end - infer_start).count();

    // 5. 计算测试误差
    std::vector<double> actual_y(test_samples);
    for (int i = 0; i < test_samples; i++)
    {
        actual_y[i] = y[train_size + i];
    }

    double mse = 0.0;
    for (int i = 0; i < test_samples; i++)
    {
        double diff = predictions[i] - actual_y[i];
        mse += diff * diff;
    }
    mse /= test_samples;

    // 反归一化显示结果
    std::vector<double> pred_denorm = predictions;
    std::vector<double> actual_denorm = actual_y;
    denormalize_data(pred_denorm, min_val, max_val);
    denormalize_data(actual_denorm, min_val, max_val);

    // 6. 输出结果
    std::cout << "\n=== CPU性能评测结果 ===" << std::endl;
    std::cout << "训练时间: " << train_time << " ms" << std::endl;
    std::cout << "推理时间: " << infer_time << " ms (" << test_samples << " 个样本)" << std::endl;
    std::cout << "平均每样本推理时间: " << infer_time / test_samples << " ms" << std::endl;
    std::cout << "推理吞吐量: " << test_samples * 1000.0 / infer_time << " 样本/秒" << std::endl;
    std::cout << "归一化MSE: " << mse << std::endl;
    std::cout << "最终训练损失: " << best_loss << std::endl;

    std::cout << "\n=== CPU预测结果对比 (前10个样本) ===" << std::endl;
    for (int i = 0; i < test_samples; i++)
    {
        std::cout << "样本 " << i + 1 << " - 预测: " << pred_denorm[i]
                  << " Mbps, 实际: " << actual_denorm[i]
                  << " Mbps, 误差: " << abs(pred_denorm[i] - actual_denorm[i]) << " Mbps" << std::endl;
    }

    std::cout << "\n=== CPU基准测试完成 ===" << std::endl;
    return 0;
}