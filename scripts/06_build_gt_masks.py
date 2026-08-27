"""构建评估用 GT 分割掩码（人工 bbox + SAM2 生成 + 人工复核）。

背景：MBDD 只提供检测框、无分割掩码；为计算 mIoU 等指标，需构建小规模
GT 掩码集。为消除"用被测系统的检测结果生成 GT"的循环偏置，这里用
【人工标注框】（MBDD 原始 bbox，即 data/splits/labels/test 下的标签）作为
SAM2 提示，而非 YOLO 预测框。人工复核修正 SAM2 的分割误差后即为 GT。

流程：
  1) 读测试集 GT bbox（YOLO 格式 cls cx cy w h，归一化）；
  2) 用 GT 框裁剪 ROI 并提示 SAM2 生成掩码（与系统后处理一致）；
  3) 拼接成全图类别掩码（0=背景，1..K=类别）；
  4) 保存掩码 PNG + 叠加可视化图 + review.html 供人工逐张复核。

复核方式：直接编辑 data/gt/masks/ 下的 PNG（或用 labelme 修正），
只保留复核通过的文件作为最终 GT。
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import detect, postprocess as pp, segment
from src.config import get_config


def parse_yolo_label(txt_path: Path, h: int, w: int) -> list[tuple[int, np.ndarray]]:
    """解析 YOLO 标签，返回 [(cls, xyxy像素坐标)]。"""
    items: list[tuple[int, np.ndarray]] = []
    if not txt_path.exists():
        return items
    for line in txt_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        cx, cy, bw, bh = (float(x) for x in parts[1:5])
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        items.append((cls, np.array([x1, y1, x2, y2], dtype=np.float32)))
    return items


def main():
    ap = argparse.ArgumentParser(description="构建 GT 掩码（人工 bbox + SAM2 + 人工复核）")
    ap.add_argument("--images", default="data/splits/images/test")
    ap.add_argument("--labels", default="data/splits/labels/test")
    ap.add_argument("--out_masks", default="data/gt/masks")
    ap.add_argument("--out_vis", default="data/gt/vis")
    ap.add_argument("--num", type=int, default=200,
                    help="构建 GT 的图像数（人工复核规模）")
    ap.add_argument("--seed", type=int, default=42,
                    help="分层抽样随机种子")
    args = ap.parse_args()

    cfg = get_config()
    sam2 = segment.build_sam2(cfg)

    img_dir = Path(args.images)
    lbl_dir = Path(args.labels)
    mask_dir = Path(args.out_masks)
    vis_dir = Path(args.out_vis)
    mask_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted(img_dir.glob("*.jpg"))
    # 只处理有标签（含目标）的图
    stems = sorted(p.stem for p in img_files if (lbl_dir / f"{p.stem}.txt").exists())
    # 分层抽样：按主类别（该图标签中出现最多的类）分组，各组均分 num，
    # 保证三类缺陷都进入 GT 集，避免按字母序取前 N 张而漏掉锈蚀等稀有类。
    rng = random.Random(args.seed)
    groups: dict[int, list[str]] = {}
    for s in stems:
        txt = (lbl_dir / f"{s}.txt").read_text(encoding="utf-8")
        classes = [int(ln.split()[0]) for ln in txt.strip().splitlines() if ln.strip()]
        if not classes:
            continue
        primary = max(set(classes), key=classes.count)
        groups.setdefault(primary, []).append(s)
    per_class = max(1, args.num // max(1, len(groups)))
    selected: list[str] = []
    for lst in groups.values():
        rng.shuffle(lst)
        selected.extend(lst[:per_class])
    rng.shuffle(selected)
    selected = selected[: args.num]
    img_files = [img_dir / f"{s}.jpg" for s in selected]
    print(f"[GT] 待生成 {len(img_files)} 张（按类别分层抽样 seed={args.seed}），"
          f"生成后请人工复核 data/gt/vis/ 下的叠加图")

    for img_path in img_files:
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        boxes = parse_yolo_label(lbl_dir / f"{img_path.stem}.txt", h, w)
        full_mask = np.zeros((h, w), dtype=np.uint8)

        for cls, xyxy in boxes:
            roi, offset = detect.crop_roi(image, xyxy, margin_ratio=0.1)
            if roi.size == 0:
                continue
            roi_box = np.array([xyxy[0] - offset[0], xyxy[1] - offset[1],
                                xyxy[2] - offset[0], xyxy[3] - offset[1]],
                               dtype=np.float32)
            roi_mask = segment.segment_roi(sam2, roi, roi_box)
            roi_mask = pp.postprocess_mask(roi_mask, cfg)
            if roi_mask.sum() == 0:
                continue
            placed = detect.place_roi_mask(roi_mask, offset, (h, w))
            full_mask[placed > 0] = cls + 1

        cv2.imwrite(str(mask_dir / f"{img_path.stem}.png"), full_mask)
        from src import pipeline as pl
        vis = pl.overlay(image, full_mask, cfg)
        cv2.imwrite(str(vis_dir / f"{img_path.stem}.jpg"), vis)

    # 生成复核索引页
    done = sorted(vis_dir.glob("*.jpg"))
    rows = "".join(
        f'<div style="display:inline-block;margin:8px">'
        f'<img src="vis/{p.stem}.jpg" width="360"/><br/>'
        f'<span>{p.stem}.png</span></div>'
        for p in done
    )
    html = f"<html><body><h2>GT 掩码复核</h2>{rows}</body></html>"
    (Path(args.out_masks).parent / "review.html").write_text(html, encoding="utf-8")
    print(f"[GT] 完成 {len(done)} 张。请打开 data/gt/review.html 复核，修正 masks/ 下的 PNG 作为最终 GT")


if __name__ == "__main__":
    main()
