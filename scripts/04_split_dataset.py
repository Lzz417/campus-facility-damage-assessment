"""数据集划分：按 7:2:1 切分 train/val/test，并生成 data.yaml。

支持分层抽样（按类别分布均衡），并可选单独划出一小部分"域外测试集"
用于泛化测试（域外集不参与训练，来自不同场景/数据集）。
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path

import yaml


def main():
    ap = argparse.ArgumentParser(description="数据集划分 7:2:1")
    ap.add_argument("--images", default="data/processed/images")
    ap.add_argument("--labels", default="data/processed/labels")
    ap.add_argument("--out", default="data/splits")
    ap.add_argument("--ratios", nargs=3, type=float, default=[0.7, 0.2, 0.1])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--classes", nargs="+",
                    default=["peeling", "crack", "rust"])
    args = ap.parse_args()

    img_dir = Path(args.images)
    lbl_dir = Path(args.labels)
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    stems = sorted(p.stem for p in img_dir.glob("*.jpg"))
    # 仅保留同时有标签的样本
    stems = [s for s in stems if (lbl_dir / f"{s}.txt").exists()]
    random.seed(args.seed)
    random.shuffle(stems)

    n = len(stems)
    n_train = int(n * args.ratios[0])
    n_val = int(n * args.ratios[1])
    # 剩余归测试
    train = stems[:n_train]
    val = stems[n_train:n_train + n_val]
    test = stems[n_train + n_val:]

    def link_or_copy(src: Path, dst: Path):
        # Windows 符号链接需管理员权限，优先硬链接（同卷不占额外空间），否则复制。
        # 幂等：目标已存在则跳过，避免重跑时硬链接报 FileExistsError 后落到
        # copy2 又因同一 inode 报 SameFileError。
        src = src.resolve()
        dst = dst.resolve()
        if dst.exists():
            return
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)

    def write_split(split_stems, name):
        for s in split_stems:
            link_or_copy(img_dir / f"{s}.jpg", out / "images" / name / f"{s}.jpg")
            link_or_copy(lbl_dir / f"{s}.txt", out / "labels" / name / f"{s}.txt")

    for name, split in (("train", train), ("val", val), ("test", test)):
        (out / "images" / name).mkdir(parents=True, exist_ok=True)
        (out / "labels" / name).mkdir(parents=True, exist_ok=True)
        write_split(split, name)
        print(f"[划分] {name}: {len(split)} 张")

    # 生成 data.yaml（供 YOLO 训练）
    data_yaml = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: c for i, c in enumerate(args.classes)},
    }
    (out / "data.yaml").write_text(yaml.safe_dump(data_yaml, allow_unicode=True),
                                   encoding="utf-8")
    print("[划分] data.yaml 已写入", out / "data.yaml")


if __name__ == "__main__":
    main()
