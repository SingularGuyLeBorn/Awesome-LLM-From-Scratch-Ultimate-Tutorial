# FILE: data_pipeline/tokenizer/teach/train_tokenizer_teach.py
"""
【教学演示脚本】
训练并测试手写的、纯Python的BPE分词器。

!!! 警告 !!!
这是一个非常非常慢的脚本，仅用于教学和算法演示。
它被设计为在极小的数据上运行，以便能在合理的时间内完成。

---
推荐用法（几分钟内完成）:
---
python data_pipeline/tokenizer/teach/train_tokenizer_teach.py
"""
import argparse
from pathlib import Path
import time
from bpe_teach import SimpleTokenizer  # 导入我们手写的教学版分词器


def main():
    parser = argparse.ArgumentParser(
        description="【教学版】训练一个手写的BPE分词器（非常慢）。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # 默认词表大小非常小，仅用于演示
    parser.add_argument("--vocab_size", type=int, default=300, help="目标词表大小 (>256)")
    # 默认只使用1MB数据，以确保能快速完成
    parser.add_argument("--data_limit_mb", type=int, default=1, help="数据量上限(MB)")
    args = parser.parse_args()

    # --- 路径定义 ---
    # 使用相对路径，确保脚本在任何位置都能正确找到文件
    data_path = Path(__file__).parent.parent.parent / "processed_data"
    train_file = data_path / "train.txt"
    output_dir = Path(__file__).parent / "toy_tokenizer_model"
    output_dir.mkdir(exist_ok=True)  # 创建保存模型的目录
    merges_file = output_dir / f"merges_vs{args.vocab_size}.txt"
    vocab_file = output_dir / f"vocab_vs{args.vocab_size}.json"

    if not train_file.exists():
        print(f"❌ 错误: 训练文件 '{train_file}' 不存在。")
        print("请先运行 'data_pipeline/processing/process_tinystories.py'。")
        return

    if merges_file.exists():
        print(f"❌ 错误: 目标模型文件 '{merges_file}' 已存在。")
        print("如果想重新训练，请先手动删除 'teach/toy_tokenizer_model' 文件夹。")
        return

    # --- 1. 训练 ---
    print("=" * 60)
    print("🎓 开始训练教学版BPE分词器")
    print(f"   词表大小: {args.vocab_size}, 数据限制: {args.data_limit_mb} MB")
    print("   (预计耗时: 1-5 分钟)")
    print("=" * 60)

    with open(train_file, 'r', encoding='utf-8') as f:
        text = f.read(args.data_limit_mb * 1024 * 1024)

    tokenizer = SimpleTokenizer()
    t0 = time.time()
    tokenizer.train(text, args.vocab_size, verbose=True)
    t1 = time.time()

    print(f"\n✅ 训练完成，总耗时 {t1 - t0:.2f} 秒。")

    # --- 2. 保存模型 ---
    # 为了清晰，我们将合并规则和词汇表分开保存
    print("\n💾 正在保存玩具模型...")

    # 保存合并规则
    with open(merges_file, 'w', encoding='utf-8') as f:
        for pair, idx in tokenizer.merges.items():
            f.write(f"{pair[0]} {pair[1]}\n")
    print(f"   - 合并规则已保存到: {merges_file}")

    # 保存词汇表 (使用JSON以便阅读)
    import json
    # bytes不能直接json序列化，需要先解码
    decoded_vocab = {k: v.decode('utf-8', errors='replace') for k, v in tokenizer.vocab.items()}
    with open(vocab_file, 'w', encoding='utf-8') as f:
        json.dump(decoded_vocab, f, ensure_ascii=False, indent=2)
    print(f"   - 词汇表已保存到: {vocab_file}")

    # --- 3. 验证 ---
    print("\n" + "=" * 60)
    print("🧪 验证玩具分词器")
    print("=" * 60)

    test_text = "Once upon a time, there was a tiny little dragon."
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)

    print(f"原文: '{test_text}'")
    print(f"编码后的Token ID序列: {encoded}")
    print(f"解码后的文本: '{decoded}'")

    assert test_text == decoded, "❌ 编解码不一致！"
    print("\n✅ 验证成功！手写分词器工作正常。")


if __name__ == "__main__":
    main()
# END OF FILE: data_pipeline/tokenizer/teach/train_tokenizer_teach.py