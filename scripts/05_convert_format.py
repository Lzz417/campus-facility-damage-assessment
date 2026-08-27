"""格式转换：掩码图 → YOLO-seg 多边形标签。

用于把 CrackForest 等像素级掩码转成 YOLO-seg 训练格式，或构建
评估所需的 GT 掩码。输出每行：class x1 y1 x2 y2 ...（归一化坐标）。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def mask_to_yolo_seg_line(mask: np.ndarray, class_id: int,
                          w: int, h: int) -> str | None:
    """提取外轮廓并转 YOLO-seg 多边形。无有效轮廓返回 None。"""
    binmask = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binmask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for cnt in contours:
        pts = cnt.reshape(-1, 2).astype(float)
        if len(pts) < 3:
            continue
        pts[:, 0] /= w
        pts[:, 1] /= h
        coords = " ".join(f"{p[0]:.6f} {p[1]:.6f}" for p in pts)
        lines.append(f"{class_id} {coords}")
    return "\n".join(lines) if lines else None


def main():
    ap = argparse.ArgumentParser(description="掩码图转 YOLO-seg 标签")
    ap.add_argument("--masks", default="data/raw/CrackForest/masks")
    ap.add_argument("--images", default="data/raw/CrackForest/images")
    ap.add_argument("--out_labels", default="data/processed/labels_seg")
    ap.add_argument("--class_id", type=int, default=1, help="映射到的类别")
    args = ap.parse_args()

    mask_dir = Path(args.masks)
    img_dir = Path(args.images)
    out = Path(args.out_labels)
    out.mkdir(parents=True, exist_ok=True)

    n = 0
    for mask_path in sorted(mask_dir.glob("*.png")):
        stem = mask_path.stem
        img_path = img_dir / f"{stem}.jpg"
        if not img_path.exists():
            img_path = img_dir / f"{stem}.png"
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        line = mask_to_yolo_seg_line(mask, args.class_id, w, h)
        if line:
            (out / f"{stem}.txt").write_text(line + "\n", encoding="utf-8")
            n += 1
    print(f"[转换] 已生成 {n} 个 YOLO-seg 标签 → {out}")


if __name__ == "__main__":
    main()
