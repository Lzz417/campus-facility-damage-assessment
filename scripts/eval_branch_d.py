"""在全部 198 张 GT 上对比「旧配置(无新特性) vs 新配置(逐类 box 边距 + 逐类面积下限)」。

目的：确认消融报告 D 组基线（mIoU 0.692 / Precision 0.835 / Recall 0.802 / F1 0.818）
能否被本脚本复现（验证评估口径一致），再量化新特性对 mIoU 的真实影响。

口径与 experiments/ablation.py 完全一致：
  - 二值缺陷 mIoU = pixel_confusion(pred>0, gt>0, nc=2) 的 iou[1]
  - 逐类 mIoU    = pixel_confusion(pred, gt, nc=4) 的 iou[1..3] 均值

用法：python scripts/eval_branch_d.py [--limit N]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config as cfg_mod
from src import detect, evaluate, pipeline, segment


def run_eval(cfg, yolo, sam2, pairs, label: str) -> dict:
    nc = 4
    tp = np.zeros(nc, np.int64)
    fp = np.zeros(nc, np.int64)
    fn = np.zeros(nc, np.int64)
    t_bin = np.zeros(2, np.int64)
    fp_bin = np.zeros(2, np.int64)
    fn_bin = np.zeros(2, np.int64)
    t0 = time.perf_counter()
    for i, (img_path, mask_path) in enumerate(pairs, 1):
        image = cv2.imread(str(img_path))
        gt = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if image is None or gt is None:
            continue
        gt = np.asarray(gt, dtype=np.uint8)
        res = pipeline.run_pipeline(image, cfg, yolo_model=yolo, sam2_predictor=sam2)
        tp2, fp2, fn2 = evaluate.pixel_confusion(res["mask"], gt, nc)
        tp += tp2
        fp += fp2
        fn += fn2
        tb, fb, nb = evaluate.pixel_confusion((res["mask"] > 0).astype(np.uint8),
                                              (gt > 0).astype(np.uint8), 2)
        t_bin += tb
        fp_bin += fb
        fn_bin += nb
        if i % 40 == 0 or i == len(pairs):
            print(f"[{label}] {i}/{len(pairs)}", flush=True)
    elapsed = time.perf_counter() - t0

    classes = [1, 2, 3]
    ious = evaluate.iou_per_class(tp, fp, fn)
    p_all, r_all, f_all = (evaluate.precision(tp, fp), evaluate.recall(tp, fn),
                           evaluate.f1_score(evaluate.precision(tp, fp), evaluate.recall(tp, fn)))
    bin_iou = evaluate.iou_per_class(t_bin, fp_bin, fn_bin)[1]
    return {
        "label": label,
        "bin_miou": float(bin_iou),
        "mean_miou": float(np.mean([ious[c] for c in classes])),
        "mean_precision": float(np.mean([p_all[c] for c in classes])),
        "mean_recall": float(np.mean([r_all[c] for c in classes])),
        "mean_f1": float(np.mean([f_all[c] for c in classes])),
        "per_class": {c: float(ious[c]) for c in classes},
        "elapsed": elapsed,
        "n": len(pairs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cfg = cfg_mod.get_config()
    bm = cfg.detect.get("box_margin")
    mp = cfg.postprocess.get("min_area_px_per_class")

    def _as_dict(o):
        return {k: o.get(k, 0) for k in ["peeling", "crack", "rust"]} if o else {}

    print(f"[对比] 新配置 box_margin={_as_dict(bm)} min_px={_as_dict(mp)}")
    yolo = detect.load_yolo(cfg)
    sam2 = segment.build_sam2(cfg)

    mask_dir = ROOT / "data" / "gt" / "masks"
    img_dir = ROOT / "data" / "splits" / "images" / "test"
    mask_files = sorted(mask_dir.glob("*.png"))
    if args.limit > 0:
        mask_files = mask_files[: args.limit]
    pairs = [(img_dir / f"{mf.stem}.jpg", mf) for mf in mask_files
             if (img_dir / f"{mf.stem}.jpg").exists()]
    print(f"[对比] 评估 {len(pairs)} 张")

    # 旧配置（特性清零）
    print("[对比] ===== 旧配置（box_margin=∅, min_px=∅）=====")
    cfg.detect.box_margin = {}
    cfg.postprocess.min_area_px_per_class = {}
    old = run_eval(cfg, yolo, sam2, pairs, "旧")

    # 仅 min_px（隔离面积过滤的收益/损失）
    cfg.detect.box_margin = {}
    cfg.postprocess.min_area_px_per_class = mp
    print("[对比] ===== 仅 min_px（box_margin=∅）=====")
    only_mp = run_eval(cfg, yolo, sam2, pairs, "min_px")

    # 新配置（恢复）
    cfg.detect.box_margin = bm
    cfg.postprocess.min_area_px_per_class = mp
    print("[对比] ===== 新配置（逐类 box 边距 + 逐类面积下限）=====")
    new = run_eval(cfg, yolo, sam2, pairs, "新")

    print("\n========== 旧 vs 仅min_px vs 新（%d 张 GT）==========" % len(pairs))
    hdr = f"{'指标':<22}{'旧配置':>12}{'仅min_px':>12}{'新配置':>12}"
    print(hdr)
    for k, label in [("bin_miou", "缺陷二值 mIoU"), ("mean_miou", "逐类均值 mIoU"),
                     ("mean_precision", "逐类均值 Precision"), ("mean_recall", "逐类均值 Recall"),
                     ("mean_f1", "逐类均值 F1")]:
        print(f"{label:<24}{old[k]:>12.4f}{only_mp[k]:>12.4f}{new[k]:>12.4f}")
    print("\n逐类 mIoU：")
    for c, cn in [(1, "peeling"), (2, "crack"), (3, "rust")]:
        print(f"  {cn:<10} 旧 {old['per_class'][c]:.4f} | 仅min_px {only_mp['per_class'][c]:.4f}"
              f" | 新 {new['per_class'][c]:.4f}")
    print(f"\n耗时：旧 {old['elapsed']:.0f}s / 仅min_px {only_mp['elapsed']:.0f}s / 新 {new['elapsed']:.0f}s")


if __name__ == "__main__":
    main()
