import os
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm


def download_tinystories():
    """
    从Hugging Face Hub下载TinyStories数据集并保存到本地。
    """
    # 定义数据集名称和本地保存路径
    dataset_name = "roneneldan/TinyStories"
    # 保存到 data_pipeline/downloaded_data/TinyStories 目录下
    save_path = Path(__file__).parent.parent / "downloaded_data" / "TinyStories"

    # 如果数据已存在，则跳过下载
    if save_path.exists():
        print(f"✅ 数据集 '{dataset_name}' 已存在于 '{save_path}'")
        return

    print(f"📥 正在从 Hugging Face Hub 下载数据集 '{dataset_name}'...")

    # 使用流式模式加载数据集以节省内存
    dataset = load_dataset(dataset_name, split='train', streaming=True)

    # 转换为可迭代的数据集以便保存
    iterable_dataset = iter(dataset)

    # 创建保存目录
    save_path.mkdir(parents=True, exist_ok=True)

    # 估算总样本数以便显示进度条 (TinyStories 约有 210 万个样本)
    total_samples = 2119719

    # 保存为 Hugging Face 的 datasets 格式
    # 注意：save_to_disk 不直接支持 streaming dataset，我们需要手动迭代
    # 这里我们采用一个变通方法：直接下载并保存
    # 对于非常大的数据集，需要更复杂的流式处理逻辑

    # 重新加载非流式数据集以便保存
    print("下载原始文件中，这可能需要一些时间...")
    full_dataset = load_dataset(dataset_name, split='train')

    print(f"💾 正在将数据集保存到 '{save_path}'...")
    full_dataset.save_to_disk(str(save_path))

    print(f"✅ 数据集下载并保存完毕。")


if __name__ == "__main__":
    download_tinystories()