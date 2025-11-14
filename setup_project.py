import os
from pathlib import Path

# --- 核心代码区结构定义 ---
code_structure = {
    "configs": [
        "model_0.1B_config.yaml", "pretrain_config.yaml", "sft_config.yaml", "dpo_config.yaml",
    ],
    "data_pipeline": {
        "download": ["download_tinystories.py"],
        "processing": ["process_tinystories.py"],
        "tokenizer": ["bpe.py", "train_tokenizer.py", "train_tokenizer_fast.py"],
    },
    "models": {
        "blocks": {
            "normalization": ["__init__.py", "normalization.py"],
            "feedforward": ["__init__.py", "feedforward.py"],
            "attention": ["__init__.py", "attention.py"],
            "positional_encoding": ["__init__.py", "positional_encoding.py"],
        },
        "__init__.py": None, "config.py": None, "transformer.py": None, "reward_model.py": None,
    },
    # ... 其他顶级目录 ...
    "pretrain": {}, "finetune": {}, "align": {}, "evaluation": {}, "inference": {}
}

# --- 文档区结构定义 ---
docs_structure = {
    "data_pipeline": ["tokenizer.md"],
    "models": {
        "blocks": ["normalization.md", "feedforward.md", "attention.md", "positional_encoding.md"],
        "transformer_architecture.md": None
    },
    "training": ["pretraining.md", "finetuning.md", "alignment.md"],
    "evaluation": ["metrics.md"],
    "inference": ["optimization.md"],
    "_media": []  # 用于存放文档中的图片
}


def create_structure(base_path, structure, create_files=True):
    """递归创建目录结构，并可选择是否创建文件。"""
    for name, content in structure.items():
        current_path = base_path / name
        if isinstance(content, dict):
            current_path.mkdir(parents=True, exist_ok=True)
            create_structure(current_path, content, create_files)
        elif isinstance(content, list) and create_files:
            current_path.mkdir(parents=True, exist_ok=True)
            for item in content:
                (current_path / item).touch()
        elif create_files and (content is None or isinstance(content, str)):
            current_path.parent.mkdir(parents=True, exist_ok=True)
            current_path.touch()


if __name__ == "__main__":
    project_root = Path(__file__).parent

    # --- 创建文档区文件夹 ---
    print("📂 正在创建 'docs' 文件夹结构...")
    docs_root = project_root / "docs"
    docs_root.mkdir(exist_ok=True)
    create_structure(docs_root, docs_structure, create_files=False)  # create_files=False 只创建目录
    (docs_root / ".gitkeep").touch()  # 添加一个.gitkeep文件，确保空目录能被git跟踪

    print("✅ 'docs' 结构创建完毕。你可以将 .md 文件移动到此处。")

    # --- (可选) 重新生成代码区结构 (如果需要的话) ---
    # print("\n📂 正在创建代码区结构...")
    # create_structure(project_root, code_structure, create_files=True)
    # print("✅ 代码区结构创建完毕。")