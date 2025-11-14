# FILE: docs/create_docs_structure.py
"""
专用脚本：仅用于在 'docs' 文件夹内部创建文档所需的**完整且深度嵌套**的子目录结构。
此脚本不创建任何文件，只创建文件夹。

用法:
1. 将此文件放置在 'docs' 文件夹内。
2. 在终端中，进入 'docs' 文件夹。
3. 运行 `python create_docs_structure.py`。
"""
from pathlib import Path

# --- 完整的、深度嵌套的文档区结构定义 ---
# 使用空字典 {} 表示一个需要创建的目录。
docs_structure = {
    "data_pipeline": {
        "download": {},
        "processing": {},
        "tokenizer": {},
    },
    "models": {
        "blocks": {
            "normalization": {},
            "feedforward": {},
            "attention": {},
            "positional_encoding": {},
        },
    },
    "training": {
        "pretraining": {},
        "finetuning": {
            "peft": {},
        },
        "alignment": {
            "algorithms": {
                "dpo": {},
                "ppo": {},
            }
        },
    },
    "evaluation": {
        "metrics": {},
        "benchmarks": {},
    },
    "inference": {
        "optimization": {},
    },
    "_media": {}  # 用于存放文档中的图片
}


def create_doc_dirs(base_path, structure):
    """
    递归地、深度地创建所有在结构中定义的目录。
    """
    for name, content in structure.items():
        current_path = base_path / name
        # 创建当前目录
        current_path.mkdir(parents=True, exist_ok=True)
        # 如果值是一个字典，说明还有子目录，继续递归
        if isinstance(content, dict):
            create_doc_dirs(current_path, content)


if __name__ == "__main__":
    # 脚本的父目录就是 'docs' 目录
    docs_root = Path(__file__).parent
    print(f"📂 正在 '{docs_root}' 内部创建完整的文档子目录结构...")

    create_doc_dirs(docs_root, docs_structure)

    print("\n✅ 深度文档目录结构创建完毕。所有二级、三级文件夹均已生成。")
    print("   现在你可以将对应的 .md 文件移动到这些文件夹中了。")

# END OF FILE: docs/create_docs_structure.py