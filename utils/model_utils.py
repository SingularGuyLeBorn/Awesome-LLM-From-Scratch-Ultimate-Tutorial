# FILE: utils/model_utils.py
"""
[新增] 模型结构分析工具
用于自动识别模型中的线性层名称，解决 LoRA/QLoRA 在 MLA/MoE 架构下 target_modules 配置困难的问题。
"""
import torch.nn as nn
from typing import List


def find_all_linear_names(model: nn.Module, freeze_vision: bool = True) -> List[str]:
    """
    自动查找模型中所有 nn.Linear 层的名称。

    策略:
    1. 递归遍历模型。
    2. 排除 lm_head (通常我们不希望微调输出层，除非特定需求)。
    3. 支持识别 MoE 专家内部的层和 MLA 的复杂投影层。
    """
    cls = nn.Linear
    lora_module_names = set()

    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            # 获取叶子节点的名称 (例如 'wq', 'w_gate', 'experts.0.w_down')
            # 我们需要的是用于匹配的后缀或全名路径

            # 排除 lm_head
            if "lm_head" in name:
                continue

            # 记录完整的模块名称
            lora_module_names.add(name)

    # 这里我们需要返回的是用于 replace_linear_with_qlora 中匹配的关键字
    # 简单的策略是：返回所有末级名称的集合 (e.g. "wq", "w_up")
    # 复杂的策略是：返回具体的路径

    # 为了兼容我们现有的 replace 逻辑 (any(t in name for t in target_modules))
    # 我们提取唯一的末级名称
    target_names = set()
    for name in lora_module_names:
        # 提取最后一部分，例如 layers.0.attention.wq -> wq
        # 但对于 MoE，可能是 experts.0.w_gate，我们希望捕获 w_gate
        parts = name.split('.')
        target_names.add(parts[-1])

    return list(target_names)


def check_architecture_compatibility(model_args, stage: str):
    """
    检查当前架构是否支持特定的训练/推理阶段。
    stage: 'sft', 'rlhf', 'inference_paged'
    """
    variant = model_args.attention_variant

    # PagedAttention 目前仅支持标准 Softmax Attention (MHA/GQA/MQA/MLA)
    # 不支持 Linear Attention 或 NSA (因为它们的 KV Cache 机制完全不同)
    if stage == 'inference_paged':
        if variant in ['linear', 'moba', 'nsa']:
            print(f"⚠️  WARNING: Architecture '{variant}' is NOT compatible with PagedAttention engine.")
            print(f"    Falling back to standard generation (non-paged) is recommended.")
            return False

    return True
# END OF FILE: utils/model_utils.py