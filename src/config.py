"""配置加载模块：读取 configs/config.yaml 并解析路径。

提供嵌套字典访问（cfg.get(...)）与便捷的属性访问，其余模块统一
`from src.config import load_config` 使用，避免散落硬编码路径。
"""
from __future__ import annotations

from pathlib import Path

import yaml

# 项目根目录（本文件位于 <root>/src/config.py）
ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "config.yaml"


def _resolve(d: dict, root: Path) -> dict:
    """递归将 paths 下的相对路径解析为相对项目根的绝对路径。"""
    for k, v in d.items():
        if isinstance(v, dict):
            _resolve(v, root)
        elif isinstance(v, str) and (k in {"raw_dir", "processed_dir",
                                            "split_dir", "weights_dir",
                                            "output_dir", "data_dir",
                                            "model", "checkpoint"}):
            # 含目录的路径解析为相对项目根的绝对路径；纯文件名(如 yolo11n.pt)保持不变
            if "/" in v or "\\" in v:
                d[k] = str(root / v)
    return d


def load_config(path: str | Path | None = None) -> dict:
    cfg_path = Path(path) if path else CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = _resolve(cfg, ROOT)
    cfg["_root"] = str(ROOT)
    return cfg


class Config:
    """属性式访问封装，cfg.classes.names 等价于 dict['classes']['names']。"""

    def __init__(self, data: dict):
        for k, v in data.items():
            if isinstance(v, dict):
                v = Config(v)
            setattr(self, k, v)

    def get(self, key, default=None):
        return getattr(self, key, default)


def get_config(path: str | Path | None = None) -> Config:
    return Config(load_config(path))


if __name__ == "__main__":
    cfg = get_config()
    print("root:", cfg._root)
    print("classes:", cfg.classes.names)
    print("detect model:", cfg.detect.model)
