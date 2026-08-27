"""为消融实验准备"原集-only"训练集（不含增强变体）。

背景：03_augment_dataset.py 把 6 类退化变体写进了 data/splits/images/train/，
使 train 目录同时包含原图与变体。消融需对比"仅原图训练 vs 原图+增强训练"，
故从 train 目录中筛出【原图】（文件名不含退化后缀）硬链接到 train_orig，
并生成 data_orig.yaml。

原图判别：变体文件名为 {stem}_{退化名}.jpg（退化名见 DEGRADATIONS），
原图为 {stem}.jpg，即 stem 不以任一退化名结尾。

输入：data/splits/images/train、data/splits/labels/train（增强后混合）
输出：data/splits/images/train_orig、labels/train_orig、data_orig.yaml
"""
from __future__ import annotations

import shutil
from pathlib import Path

import yaml

SUFFIXES = ("_dark", "_backlight", "_fog", "_shadow", "_occlusion", "_blur")


def copy(src: Path, dst: Path) -> None:
    """复制文件（不用硬链接）。

    实测：在慢盘 HDD（ST2000DM005）上 os.link 硬链接会间歇性卡死，
    每个进程恰好 4 次即挂、无报错退出。shutil.copy2 顺序读写稳定可靠。
    """
    src = src.resolve()
    dst = dst.resolve()
    if dst.exists():
        return
    shutil.copy2(src, dst)


def main():
    root = Path("data/splits")
    img_dir = root / "images" / "train"
    lbl_dir = root / "labels" / "train"
    out_img = root / "images" / "train_orig"
    out_lbl = root / "labels" / "train_orig"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    orig_stems = sorted(p.stem for p in img_dir.glob("*.jpg")
                        if not p.stem.endswith(SUFFIXES))
    print(f"[原集] 筛出原图 {len(orig_stems)} 张（train 共 "
          f"{len(list(img_dir.glob('*.jpg')))} 张）")

    total = len(orig_stems)
    for i, s in enumerate(orig_stems, 1):
        copy(img_dir / f"{s}.jpg", out_img / f"{s}.jpg")
        copy(lbl_dir / f"{s}.txt", out_lbl / f"{s}.txt")
        if i % 500 == 0 or i == total:
            print(f"[原集] 已复制 {i}/{total}", flush=True)

    data_orig = {
        "path": str(root.resolve()),
        "train": "images/train_orig",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "peeling", 1: "crack", 2: "rust"},
    }
    (root / "data_orig.yaml").write_text(
        yaml.safe_dump(data_orig, allow_unicode=True), encoding="utf-8")
    print("[原集] 完成。data_orig.yaml 已写入，"
          f"train_orig 共 {len(list(out_img.glob('*.jpg')))} 张")


if __name__ == "__main__":
    main()
