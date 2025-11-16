# FILE: utils/config_loader.py
"""
[新增] 专职的配置加载模块。
"""
import yaml
from pathlib import Path
from types import SimpleNamespace

def load_config(config_path: str, project_base_path: Path) -> SimpleNamespace:
    """
    加载YAML配置文件，解析相对路径，并递归转换为SimpleNamespace对象。
    """
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    def resolve_paths_in_dict(d: dict, base: Path):
        if not isinstance(d, dict):
            return
        for k, v in d.items():
            if isinstance(v, dict):
                resolve_paths_in_dict(v, base)
            elif isinstance(v, str) and ("_dir" in k or "_path" in k or "tokenizer_name" in k or "load_from_checkpoint" in k):
                if v.startswith('./'):
                    # 将相对于项目的路径转换为绝对路径
                    d[k] = str((base / v[2:]).resolve())

    resolve_paths_in_dict(config_dict, project_base_path)

    def dict_to_sns(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_sns(v) for k, v in d.items()})
        return d

    return dict_to_sns(config_dict)
# END OF FILE: utils/config_loader.py