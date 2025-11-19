"""师生 EMA 半监督模型实现。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from scheduler.hparam_scheduler import HParams


class MLP(nn.Module):
    """简单的多层感知机，支持 LayerNorm。"""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[list[int]] = None,
        num_classes: int = 2,
        dropout: float = 0.0,
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [64, 64]
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            if use_layernorm:
                layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


@dataclass
class LossOutputs:
    """封装损失。"""

    supervised: Tensor
    unsupervised: Tensor
    total: Tensor
    details: Dict[str, Tensor]


class TeacherStudentModel:
    """维护学生与教师网络及损失计算。"""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: Optional[list[int]] = None,
        dropout: float = 0.0,
        device: Optional[torch.device] = None,
        use_layernorm: bool = True,
    ) -> None:
        self.student = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_classes=num_classes,
            dropout=dropout,
            use_layernorm=use_layernorm,
        )
        self.teacher = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_classes=num_classes,
            dropout=dropout,
            use_layernorm=use_layernorm,
        )
        self.device = device or torch.device("cpu")
        self.to(self.device)
        self._init_teacher()

    def to(self, device: torch.device) -> None:
        """移动到指定设备。"""
        self.device = device
        self.student.to(device)
        self.teacher.to(device)

    def _init_teacher(self) -> None:
        self.teacher.load_state_dict(self.student.state_dict())
        for param in self.teacher.parameters():
            param.requires_grad_(False)

    def forward_student(self, x: Tensor) -> Tensor:
        return self.student(x)

    def forward_teacher(self, x: Tensor) -> Tensor:
        return self.teacher(x)

    def compute_losses(
        self,
        x_labeled: Optional[Tensor],
        y_labeled: Optional[Tensor],
        x_unlabeled: Optional[Tensor],
        hparams: HParams,
    ) -> LossOutputs:
        """计算监督与非监督损失。"""
        device = self.device
        sup_loss = torch.zeros((), device=device)
        details: Dict[str, Tensor] = {}
        if x_labeled is not None and x_labeled.numel() > 0:
            logits = self.forward_student(x_labeled)
            sup_loss = F.cross_entropy(logits, y_labeled)
            details["student_logits_labeled"] = logits.detach()
        unsup_loss = torch.zeros((), device=device)
        hard_loss = torch.zeros((), device=device)
        soft_loss = torch.zeros((), device=device)
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
        details["hard_loss"] = hard_loss.detach()
        details["soft_loss"] = soft_loss.detach()
        return LossOutputs(
            supervised=sup_loss,
            unsupervised=unsup_loss,
            total=total_loss,
            details=details,
        )

    def update_teacher(self, alpha: float) -> None:
        """EMA 更新教师参数。"""
        with torch.no_grad():
            for teacher_param, student_param in zip(
                self.teacher.parameters(),
                self.student.parameters(),
            ):
                teacher_param.data.mul_(alpha).add_(
                    student_param.data * (1.0 - alpha)
                )
