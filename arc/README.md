# EuroSAT MLP 作业代码说明

这个目录保存了本次作业的完整代码：用 `NumPy` 从零实现一个 MLP 分类器，在 `EuroSAT_RGB` 遥感图像数据集上完成训练、超参数搜索、测试评估、可视化分析，以及调用大模型分析误分类样本。

## 1. 目录结构

| 路径 | 作用 |
| --- | --- |
| `train.py` | 单次训练入口，用指定超参数训练一个 MLP，并保存最优模型和可视化结果。 |
| `search.py` | 超参数搜索入口，支持 grid search 和 random search，会为每组参数生成一个 `run_xx` 目录。 |
| `evaluate.py` | 模型评估入口，可以评估单个 checkpoint，也可以批量评估多个搜索目录下的模型。 |
| `requirements.txt` | 运行代码需要的 Python 依赖。 |
| `mlp_hw1/` | MLP、自动微分、数据加载、训练、优化器、指标和可视化等核心实现。 |
| `llm/` | 调用多模态大模型分析误分类样本的代码。 |
| `artifacts/` | 训练、搜索、评估和报告用图片等输出结果。 |

## 2. 核心代码文件

`mlp_hw1/autograd.py` 实现了简化版自动微分系统，核心是 `Tensor` 类。前向传播时，每一步运算都会被记录成计算图；训练时调用 `loss.backward()`，程序会沿计算图反向传播并计算梯度。

`mlp_hw1/model.py` 实现了模型结构，包括 `Linear` 全连接层和 `MLPClassifier`。本作业中的三层 MLP 可以理解为 `输入层 -> 隐藏层 -> 输出层`，代码中对应 `fc1 -> activation -> fc2`。

`mlp_hw1/data.py` 负责读取 `EuroSAT_RGB` 数据集、建立缓存、划分训练集/验证集/测试集，并生成 batch。输入模型前，每张 `64x64x3` 图片会被归一化并展平成长度为 `12288` 的向量。

`mlp_hw1/trainer.py` 是训练主流程，包括前向传播、交叉熵损失、L2 正则、反向传播、SGD 参数更新、验证集选最优模型，以及最终测试集评估和图片可视化。

`mlp_hw1/visualization.py` 负责生成训练曲线、混淆矩阵、第一层权重可视化和误分类样本图。

## 3. 环境准备

建议在 `arc` 目录下安装依赖：

```bash
pip install -r requirements.txt
```

默认数据集路径是：

```text
D:\project\深度学习与空间智能\EuroSAT_RGB
```

如果数据集放在其他位置，可以在运行命令时加 `--dataset-root` 指定。

## 4. 单次训练

最简单的训练命令：

```bash
python train.py
```

使用最终实验中表现较好的配置训练：

```bash
python train.py --epochs 40 --learning-rate 0.05 --hidden-dim 256 --weight-decay 0.0002 --activation tanh --output-dir artifacts/train_final
```

训练结束后，输出目录中通常会包含：

| 文件 | 作用 |
| --- | --- |
| `best_model.npz` | 验证集准确率最高的模型参数 checkpoint。 |
| `summary.json` | 本次训练的配置、最佳 epoch、验证集准确率和测试集准确率。 |
| `history.json` | 每个 epoch 的训练/验证 loss 和 accuracy。 |
| `training_curves.png` | 训练曲线图。 |
| `confusion_matrix.png` | 测试集混淆矩阵。 |
| `first_layer_weights.png` | 第一层权重可视化。 |
| `misclassified_examples.png` | 部分误分类样本展示。 |

## 5. 超参数搜索

`search.py` 用来一次训练多组超参数。grid search 会遍历所有组合：

```bash
python search.py --mode grid --epochs 40 --learning-rates 0.05 --hidden-dims 128,256 --weight-decays 0.00005,0.0001,0.0002 --activations tanh --output-dir artifacts/output_search/search_5
```

random search 会从所有组合中随机抽取若干组：

```bash
python search.py --mode random --epochs 40 --learning-rates 0.05 --hidden-dims 256 --weight-decays 0.00005,0.0001,0.0002 --activations tanh --num-random-runs 3 --output-dir artifacts/output_search/search_7
```

每次搜索会生成多个 `run_xx` 子目录。每个 `run_xx` 都是一组超参数训练出来的完整实验结果，里面会有模型参数、训练曲线、混淆矩阵和误分类样本图。搜索目录下还会生成：

| 文件 | 作用 |
| --- | --- |
| `search_results.csv` | 所有 run 的超参数和结果汇总，按验证集准确率排序。 |
| `best_config.json` | 当前搜索中验证集表现最好的配置。 |

本次报告中最终采用的是 `epoch=40` 下验证集最优的模型，即：

```text
artifacts/output_search/search_5/run_05
```

## 6. 模型评估

评估单个模型：

```bash
python evaluate.py --checkpoint artifacts/best_model/best_model.npz --output-dir artifacts/best_model/evaluation
```

批量评估多个搜索目录：

```bash
python evaluate.py --checkpoint-dir artifacts/output_search/search_4 --checkpoint-dir artifacts/output_search/search_5 --checkpoint-dir artifacts/output_search/search_6 --checkpoint-dir artifacts/output_search/search_7 --output-dir artifacts/batch_evaluation
```

如果希望批量评估时也给每个模型生成混淆矩阵和权重图，可以加：

```bash
--save-artifacts
```

评估结果中最重要的是 `test_accuracy` 和 `test_loss`。其中 `test_accuracy` 表示测试集分类准确率，`test_loss` 表示测试集交叉熵损失；混淆矩阵可以进一步观察哪些类别容易互相混淆。

## 7. 最终模型结果

最终模型已经整理到：

```text
artifacts/best_model
```

其中最重要的文件包括：

| 文件或目录 | 作用 |
| --- | --- |
| `best_model.npz` | 最终模型参数。 |
| `best_model_summary.json` | 最终模型的配置、测试结果、类别准确率和主要混淆对。 |
| `misclassified_samples.csv` | 随机抽取的 100 条误分类样本信息，包括真实标签、预测标签、置信度和图片路径。 |
| `misclassified_samples/` | 100 张误分类图片。 |
| `llm_analysis/` | 大模型对误分类样本的分析结果。 |

当前最终模型的主要结果为：

```text
best_val_accuracy = 0.5812
test_accuracy = 0.5859
test_loss = 1.2129
```

## 8. LLM 错误样本分析

`llm/` 文件夹用于调用多模态大模型分析 MLP 为什么会把某些图片分类错。

| 文件 | 作用 |
| --- | --- |
| `config.json` | API 配置，包括模型名、base_url、temperature 等。 |
| `openai_client.py` | 通用 API 调用封装，支持图片输入，并将图片 `detail` 设置为 `high`。 |
| `analyze_misclassifications.py` | 误分类样本整理、调用大模型分析、统计结果汇总的主脚本。 |

如果只想准备最终模型目录和 100 条误分类样本，可以运行：

```bash
python llm/analyze_misclassifications.py --prepare-only
```

如果已经准备好样本，并希望调用大模型分析全部 100 条：

```bash
python llm/analyze_misclassifications.py --skip-prepare
```

如果只想先测试几条，避免一次性消耗太多 API 额度：

```bash
python llm/analyze_misclassifications.py --skip-prepare --limit 5
```

分析完成后，结果会保存在：

```text
artifacts/best_model/llm_analysis
```

其中 `analysis_summary.json` 是统计汇总，`analysis_results.csv` 是每张错误样本的具体分析结果。

## 9. 推荐工作流

如果从头复现实验，可以按下面顺序运行：

1. 安装依赖：`pip install -r requirements.txt`
2. 先跑一次单模型训练，确认环境正常：`python train.py --max-samples-per-class 50 --epochs 1`
3. 做超参数搜索：运行 `search.py`
4. 按验证集准确率选择最佳模型
5. 用 `evaluate.py` 在测试集上评估最终模型
6. 整理误分类样本并用 `llm/analyze_misclassifications.py` 做错误原因分析
7. 将 `artifacts/best_model` 中的图和统计结果写入报告

