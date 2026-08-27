"""定量评估「逐类 box 边距 + 逐类面积下限」的 before/after 效果。

在含 GT 掩码的测试图上，分别用：
  - after  = 当前 config（box_margin + min_area_px_per_class）
  - before = 两者清零（模拟旧行为）
跑同一批图，统计：
  - 各级检测数（YOLO 框数 → 保留掩码数 → 丢弃数）
  - 总/各类掩码像素
  - 每类平均掩码面积（box 扩张是否让面病害掩码更完整）
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config as cfg_mod
from src import detect as det
from src import pipeline as pl
from src import segment as seg


def main():
    cfg = cfg_mod.get_config()
    yolo = det.load_yolo(cfg)
    sam2 = seg.build_sam2(cfg)

    # 从 GT 目录挑含各类的样本（每类 ≤ 12）
    gt_dir = ROOT / "data" / "gt" / "masks"
    img_dir = ROOT / "data" / "splits" / "images" / "test"
    want = {"peeling": 1, "crack": 2, "rust": 3}
    picks: dict[str, list[Path]] = {k: [] for k in want}
    for mf in sorted(gt_dir.glob("*.png")):
        m = cv2.imread(str(mf), cv2.IMREAD_UNCHANGED)
        if m is None:
            continue
        img = img_dir / f"{mf.stem}.jpg"
        if not img.exists():
            continue
        for name, cid in want.items():
            if len(picks[name]) < 12 and (m == cid).sum() > 500:
                picks[name].append(img)

    imgs = sorted({p for lst in picks.values() for p in lst})
    print(f"[评估] 测试图 {len(imgs)} 张（GT 含该类即入选）", flush=True)

    # 先捕获原始配置，避免 run() 的原地赋值把 after 组读成空配置
    orig_bm = cfg.detect.get("box_margin")
    orig_mp = cfg.postprocess.get("min_area_px_per_class")

    def run(box_margin, min_px):
        cfg.detect.box_margin = box_margin or {}
        cfg.postprocess.min_area_px_per_class = min_px or {}
        n_det, n_keep, px, by_cls = 0, 0, 0, {k: [0, 0] for k in want}  # [像素, 检测数]
        for ip in imgs:
            image = cv2.imread(str(ip))
            res = pl.run_pipeline(image, cfg, yolo_model=yolo, sam2_predictor=sam2)
            for r in res["results"]:
                n_det += 1
                n_keep += 1
                px += r["pixel_area"]
                c = r["class"]
                if c in by_cls:
                    by_cls[c][0] += r["pixel_area"]
                    by_cls[c][1] += 1
        return n_det, n_keep, px, by_cls

    print("[评估] 跑 before（margin=0, min_px=0）...", flush=True)
    b_det, b_keep, b_px, b_cls = run({}, {})
    print("[评估] 跑 after（当前配置）...", flush=True)
    a_det, a_keep, a_px, a_cls = run(orig_bm, orig_mp)

    print("\n===== before vs after（%d 张测试图）=====" % len(imgs))
    print(f"{'指标':<28}{'before':>12}{'after':>12}")
    print(f"{'YOLO 检测框数(保留掩码)':<26}{b_keep:>12}{a_keep:>12}")
    print(f"{'总掩码像素':<28}{b_px:>12}{a_px:>12}")
    print(f"{'平均每检测掩码像素':<26}{b_px/max(b_keep,1):>12.0f}{a_px/max(a_keep,1):>12.0f}")
    for k in want:
        bb, ab = b_cls[k], a_cls[k]
        bmean = bb[0] / max(bb[1], 1)
        amean = ab[0] / max(ab[1], 1)
        print(f"  {k:<24} 检测数 {bb[1]:>4}->{ab[1]:<4} 均值 {bmean:>8.0f}->{amean:>8.0f} px")


if __name__ == "__main__":
    main()
