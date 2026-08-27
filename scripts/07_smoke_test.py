"""SAM2 + 后处理 冒烟测试：用测试集真实 bbox 直接跑 SAM2 分割，验证环境与分割链。

不依赖 YOLO 检测（yolo11n 预训练权重未适配缺陷类），仅验证：
  1) SAM2 模型能否在 GPU 上加载 2.4GB 权重；
  2) box 提示分割 + 自研后处理 整条链能否跑通并产出合理掩码。

用法：
  python scripts/07_smoke_test.py --num 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import detect, postprocess as pp, segment
from src.config import get_config


def parse_first_box(txt_path: Path, h: int, w: int) -> tuple[int, np.ndarray] | None:
    if not txt_path.exists():
        return None
    for line in txt_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        cx, cy, bw, bh = (float(x) for x in parts[1:5])
        x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
        x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
        return cls, np.array([x1, y1, x2, y2], dtype=np.float32)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/splits/images/test")
    ap.add_argument("--labels", default="data/splits/labels/test")
    ap.add_argument("--num", type=int, default=3)
    args = ap.parse_args()

    cfg = get_config()
    print("[冒烟] 构建 SAM2（首次加载 2.4GB 权重，约 30-60s）...")
    sam2 = segment.build_sam2(cfg)
    print("[冒烟] SAM2 就绪")

    img_dir = Path(args.images)
    lbl_dir = Path(args.labels)
    tested = 0
    for img_path in sorted(img_dir.glob("*.jpg")):
        box = None
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        box = parse_first_box(lbl_dir / f"{img_path.stem}.txt", h, w)
        if box is None:
            continue
        cls, xyxy = box
        roi, offset = detect.crop_roi(image, xyxy, margin_ratio=0.1)
        roi_box = np.array([xyxy[0] - offset[0], xyxy[1] - offset[1],
                            xyxy[2] - offset[0], xyxy[3] - offset[1]], dtype=np.float32)
        raw = segment.segment_roi(sam2, roi, roi_box)
        out = pp.postprocess_mask(raw, cfg)
        print(f"[冒烟] {img_path.name} 类别={cls} ROI={roi.shape[:2]} "
              f"原始前景={int(raw.sum())}px 后处理={int(out.sum())}px")
        tested += 1
        if tested >= args.num:
            break

    if tested == 0:
        print("[冒烟] 未找到带标签的测试图，检查 test 标签目录")
    else:
        print(f"[冒烟] 完成 {tested} 张，SAM2 + 后处理链路正常")


if __name__ == "__main__":
    main()
