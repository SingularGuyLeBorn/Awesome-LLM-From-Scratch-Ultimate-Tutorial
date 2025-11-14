# FILE: data_pipeline/tokenizer/train_tokenizer.py
"""
【项目主分词器训练脚本】
使用HuggingFace Tokenizers库训练BPE分词器（Rust内核，速度极快）。

这个脚本将用于训练我们项目中实际使用的分词器。
它经过了增强，加入了SFT（指令微调）阶段必需的特殊词元。

---
用法示例:
---

1.  ✅ 快速测试 (4k词表, 20MB数据):
    # 用于快速验证流程是否跑通
    python data_pipeline/tokenizer/train_tokenizer.py --vocab_size 4096 --data_limit_mb 20

2.  🔥 SFT增强版训练 (推荐, 10k词表, 200MB数据):
    # 为后续的SFT和DPO阶段做准备，生成一个高质量的分词器
    # 耗时约几分钟
    python data_pipeline/tokenizer/train_tokenizer.py --vocab_size 10000 --data_limit_mb 200

3.   marathon 完整预训练版 (16k词表, 全部数据):
    # 如果要从零开始进行大规模预训练，可以使用此配置
    # !!! 警告: 会消耗大量内存和较长时间 !!!
    python data_pipeline/tokenizer/train_tokenizer.py --vocab_size 16384 --data_limit_mb 0
"""

import argparse
from pathlib import Path
import time
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
import sys

# 将项目根目录添加到路径，以便导入utils
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from utils.file_utils import create_subset_file


def main():
    parser = argparse.ArgumentParser(
        description="【主脚本】使用HuggingFace Tokenizers训练一个用于项目的高性能BPE分词器。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--vocab_size", type=int, default=10000, help="目标词表大小")
    parser.add_argument("--data_limit_mb", type=int, default=200, help="用于训练的数据量上限(MB)，0表示无限制")
    args = parser.parse_args()

    # --- 路径设置 ---
    data_path = Path(__file__).parent.parent / "processed_data"
    train_file = data_path / "train.txt"
    output_file = data_path / f"tinystories_project_vs{args.vocab_size}.json"

    if not train_file.exists():
        print(f"❌ 错误: '{train_file}' 不存在")
        print("请先运行 'data_pipeline/processing/process_tinystories.py'")
        return

    if output_file.exists():
        print(f"❌ 模型已存在: {output_file}")
        print("如果你想重新训练，请先手动删除该文件，或指定一个不同的 vocab_size。")
        return

    print("=" * 80)
    print(f"🚀 开始训练项目主分词器 (目标词表: {args.vocab_size})")
    print("=" * 80)

    # --- 准备训练数据 ---
    temp_file = None
    if args.data_limit_mb > 0:
        print(f"📚 正在创建 {args.data_limit_mb}MB 的数据子集用于训练...")
        temp_file = data_path / "temp_train_subset.txt"
        create_subset_file(train_file, temp_file, args.data_limit_mb)
        print(f"   -> 已创建临时子集文件: {temp_file.name} ({temp_file.stat().st_size / 1e6:.2f} MB)")
        train_files = [str(temp_file)]

    else:
        print(f"📚 使用完整训练数据: {train_file.stat().st_size / 1e6:.1f}MB")
        train_files = [str(train_file)]

    # --- 1. 初始化分词器模型 ---
    print("\n1/3: 初始化BPE模型...")
    tokenizer = Tokenizer(models.BPE())

    # --- 2. 配置预分词器和解码器 ---
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    # --- 3. 定义训练器和特殊词元 ---
    print(f"2/3: 配置训练器及特殊词元...")
    special_tokens = [
        "<|endoftext|>",
        "<|pad|>",
        "<|im_start|>",
        "<|im_end|>",
    ]

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=2,
        special_tokens=special_tokens,
        show_progress=True
    )

    # --- 4. 开始训练 ---
    print(f"3/3: 开始训练...")
    t0 = time.time()
    tokenizer.train(train_files, trainer)
    t1 = time.time()

    print(f"\n✅ 训练完成！总耗时 {t1 - t0:.2f} 秒")

    # --- 5. 保存模型 ---
    tokenizer.save(str(output_file))
    print(f"💾 分词器模型已保存到: {output_file}")

    # 清理临时文件
    if temp_file and temp_file.exists():
        temp_file.unlink()
        print(f"🗑️ 已删除临时文件: {temp_file.name}")

    # --- 6. 验证 ---
    print("\n" + "=" * 80)
    print("🧪 验证分词器功能")
    print("=" * 80)

    loaded_tokenizer = Tokenizer.from_file(str(output_file))
    test_text_simple = "This is a test sentence."
    encoding_simple = loaded_tokenizer.encode(test_text_simple)
    print(f"普通文本: '{test_text_simple}'")
    print(f"  -> Tokens: {encoding_simple.tokens}")
    print(f"  -> 解码: '{loaded_tokenizer.decode(encoding_simple.ids)}'")
    assert test_text_simple == loaded_tokenizer.decode(encoding_simple.ids)

    print("-" * 40)

    test_text_special = "<|im_start|>Hello<|im_end|><|endoftext|>"
    encoding_special = loaded_tokenizer.encode(test_text_special)
    decoded_special = loaded_tokenizer.decode(encoding_special.ids, skip_special_tokens=False)

    print(f"带特殊词元的文本: '{test_text_special}'")
    print(f"  -> Tokens: {encoding_special.tokens}")
    print(f"  -> 解码 (保留特殊词元): '{decoded_special}'")
    assert test_text_special == decoded_special

    print("\n✅ 验证通过！分词器工作正常，且能正确处理特殊词元。")
    print(f"📊 最终词表大小: {loaded_tokenizer.get_vocab_size()}")


if __name__ == "__main__":
    main()
# END OF FILE: data_pipeline/tokenizer/train_tokenizer.py