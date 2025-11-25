# FILE: utils/model_utils.py
"""
[v1.1 - Strict Check] 模型结构分析工具
- [新增] check_architecture_compatibility: 明确拦截 MLA 架构运行 PagedAttention。
- 自动识别模型中的线性层名称，解决 LoRA/QLoRA 在 MLA/MoE 架构下 target_modules 配置困难的问题。
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
    # 兼容标准 Linear 和 量化 Linear (Linear4bit)
    # 注意：这里我们只检测 standard nn.Linear，因为 QLoRA 替换发生在检测之后
    # 如果已经是 QLoRA 模型，结构已经变了，但此函数通常用于转换前
    linear_cls = (nn.Linear,)

    lora_module_names = set()

    for name, module in model.named_modules():
        if isinstance(module, linear_cls):
            # 排除 lm_head
            if "lm_head" in name:
                continue

            # 记录完整的模块名称
            lora_module_names.add(name)

    # 提取唯一的末级名称
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

    # PagedAttention 目前仅支持标准 Softmax Attention (MHA/GQA/MQA)
    # [核心修改] MLA (Multi-Head Latent Attention) 由于 KV 结构不同 (Latent Compressed)，
    # 目前的 BlockManager 和 PagedAttention Kernel 尚未适配。
    # Linear/MoBA/NSA 也是同理。
    if stage == 'inference_paged':
        if variant in ['linear', 'moba', 'nsa', 'mla']:
            print(f"\n❌ [架构不兼容警告]")
            print(f"   当前架构 '{variant.upper()}' 尚未适配 PagedAttention 推理引擎 (api_server.py)。")
            print(f"   原因: 该架构使用了非标准的 KV Cache 结构 (如潜变量压缩或 RNN 状态)。")
            print(f"   建议: 请使用标准推理脚本 'inference/chat.py' 进行交互。")
            return False

    return True
# END OF FILE: utils/model_utils.py