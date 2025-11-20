# FILE: finetune/distillation/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """
    知识蒸馏损失函数。
    Loss = (1 - alpha) * CE_Loss(student, target) + alpha * (T^2) * KL_Div(student, teacher)
    """

    def __init__(self, temperature: float = 2.0, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        # KLDivLoss 默认是 mean reduction，但我们需要 batchmean 以符合数学定义
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            student_logits: (B, Seq, Vocab)
            teacher_logits: (B, Seq, Vocab)
            labels: (B, Seq)
        """
        # 1. 标准交叉熵损失 (Hard Label Loss)
        # 展平以便计算
        vocab_size = student_logits.size(-1)
        ce_loss = F.cross_entropy(
            student_logits.view(-1, vocab_size),
            labels.view(-1),
            ignore_index=-1
        )

        # 2. 蒸馏损失 (Soft Target Loss)
        # 只有在 alpha > 0 时才计算，节省计算量
        if self.alpha > 0:
            # 温度缩放
            s_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
            t_soft = F.softmax(teacher_logits / self.temperature, dim=-1)

            # 计算 KL 散度
            # 注意: KLDivLoss 期望 input 是 log-probabilities，target 是 probabilities
            # 我们只在有效 token 上计算（虽然 PyTorch 的 KLDiv 不支持 ignore_index，但通常 soft targets 包含的信息比 hard label 丰富）
            # 为了简化，我们这里对所有 token 计算，或者你可以手动 mask。
            # 为了严谨，我们使用 mask 过滤掉 padding

            mask = (labels != -1).view(-1)
            s_soft_flat = s_soft.view(-1, vocab_size)[mask]
            t_soft_flat = t_soft.view(-1, vocab_size)[mask]

            kd_loss = self.kl_div(s_soft_flat, t_soft_flat) * (self.temperature ** 2)
        else:
            kd_loss = torch.tensor(0.0, device=student_logits.device)

        # 3. 组合
        total_loss = (1 - self.alpha) * ce_loss + self.alpha * kd_loss

        return total_loss, ce_loss, kd_loss
# END OF FILE: finetune/distillation/loss.py