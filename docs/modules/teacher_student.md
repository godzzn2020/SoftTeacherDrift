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

## 关键代码片段（参考）

- MLP 结构与 Teacher–Student 定义（`models/teacher_student.py`）：

  ```python
  class MLP(nn.Module):
      def __init__(self, input_dim, hidden_dims=None, num_classes=2,
                   dropout=0.0, use_layernorm=True):
          ...
          for h in hidden_dims or [64, 64]:
              layers.append(nn.Linear(prev_dim, h))
              if use_layernorm:
                  layers.append(nn.LayerNorm(h))
              layers.append(nn.ReLU())
              if dropout > 0:
                  layers.append(nn.Dropout(dropout))
              prev_dim = h
          layers.append(nn.Linear(prev_dim, num_classes))
          self.net = nn.Sequential(*layers)

  class TeacherStudentModel:
      def __init__(..., hidden_dims=None, dropout=0.0, device=None, use_layernorm=True):
          self.student = MLP(input_dim, hidden_dims, num_classes, dropout, use_layernorm)
          self.teacher = MLP(input_dim, hidden_dims, num_classes, dropout, use_layernorm)
          self.to(device or torch.device("cpu"))
          self._init_teacher()
  ```

- 损失计算与 EMA 更新：

  ```python
  def compute_losses(self, x_labeled, y_labeled, x_unlabeled, hparams: HParams) -> LossOutputs:
      sup_loss = 0
      if x_labeled is not None and x_labeled.numel() > 0:
          logits = self.forward_student(x_labeled)
          sup_loss = F.cross_entropy(logits, y_labeled)
          details["student_logits_labeled"] = logits.detach()

      unsup_loss = hard_loss = soft_loss = 0
      if x_unlabeled is not None and x_unlabeled.numel() > 0:
          with torch.no_grad():
              teacher_logits = self.forward_teacher(x_unlabeled)
              teacher_probs = torch.softmax(teacher_logits, dim=1)
          student_logits = self.forward_student(x_unlabeled)
          student_probs = torch.softmax(student_logits, dim=1)
          max_probs, pseudo_labels = teacher_probs.max(dim=1)
          confident_mask = max_probs >= hparams.tau
          if confident_mask.any():
              hard_loss = F.cross_entropy(
                  student_logits[confident_mask],
                  pseudo_labels[confident_mask],
              )
          soft_loss = F.kl_div(
              input=torch.log_softmax(student_logits, dim=1),
              target=teacher_probs.detach(),
              reduction="batchmean",
          )
          unsup_loss = hard_loss + soft_loss
          details["teacher_probs"] = teacher_probs.detach()
          details["student_probs_unlabeled"] = student_probs.detach()

      total_loss = sup_loss + hparams.lambda_u * unsup_loss

  def update_teacher(self, alpha: float) -> None:
      with torch.no_grad():
          for t_param, s_param in zip(self.teacher.parameters(), self.student.parameters()):
              t_param.data.mul_(alpha).add_(s_param.data * (1.0 - alpha))
  ```

## 当前算法流程（Teacher–Student 部分）

1. 使用 `FeatureVectorizer` / `LabelEncoder` 将批次转换为张量：
   - 有标签部分：`(x_labeled, y_labeled)`；
   - 无标签部分：`x_unlabeled`（以及可选的 `y_unlabeled_true` 仅用于评估）。
2. 学生网络在有标签样本上计算监督交叉熵损失。
3. 教师网络对无标签样本前向，生成 `teacher_probs`：
   - 根据 `tau` 过滤出高置信度伪标签（硬标签）；
   - 同时用 KL 散度约束 `student_probs` 逼近 `teacher_probs`。
4. 将硬伪标签损失和一致性 KL 按 `lambda_u` 加权，加到总损失中。
5. 用总损失对学生更新；随后用 `alpha` 对教师做 EMA 滤波。
6. 将 `teacher_probs`、`student_probs_unlabeled`、`student_logits_labeled` 等细节通过 `LossOutputs.details` 返回，供漂移信号与评估使用。

## TODO

- 考虑加入特征归一化/嵌入层，增强对混合类型数据的适应性。
- 引入不对称一致性损失或温度调度，以改善漂移后“冷启动”表现。
- 将教师输出的熵/散度等统计直接集成至 `LossOutputs`，减少训练循环对内部细节的依赖。
