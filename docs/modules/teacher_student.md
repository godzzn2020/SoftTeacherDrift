# 教师–学生模型

## 简介

`models/teacher_student.py` 提供共享架构的学生/教师网络、损失计算与 EMA 更新逻辑，是整个半监督框架的核心。

## 模型结构

- **Student MLP**
  - 输入：定长浮点向量（由 `FeatureVectorizer` 提供）。
  - 层次：`Linear -> LayerNorm -> ReLU -> (Dropout)` 重复多次，末层 `Linear` 输出类别 logits。
  - `hidden_dims`、`dropout`、`use_layernorm` 可配置（默认 LayerNorm 以支持单样本 batch）。
- **Teacher**
  - 与学生同构但不参与梯度更新；
  - 使用参数 EMA：`theta_teacher = alpha * theta_teacher + (1 - alpha) * theta_student`。

## 损失与超参数

- `compute_losses` 同时输出：
  - **监督损失**：交叉熵 `CE(student(x_labeled), y_labeled)`。
  - **无监督损失**：由两部分组成
    - 硬伪标签：当教师预测最大概率 ≥ `tau` 时，将 argmax 作为伪标签；
    - 一致性正则：`KL(teacher_probs || student_probs)`。
  - 总损失：`L_total = L_sup + lambda_u * (L_hard + L_KL)`。
- 关键超参数：
  - `alpha`：EMA 的衰减系数；
  - `lambda_u`：无监督损失权重；
  - `tau`：伪标签置信度阈值；
  - 与训练循环共享的 `lr`、`dropout`、`hidden_dims` 等。

## TODO

- 考虑加入特征归一化/嵌入层，增强对混合类型数据的适应性。
- 引入不对称一致性损失或温度调度，以改善漂移后“冷启动”表现。
- 将教师输出的熵/散度等统计直接集成至 `LossOutputs`，减少训练循环对内部细节的依赖。

