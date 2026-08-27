"""消融对比实验（创新点验证，竞赛必看）。

固定测试集 + GT 掩码，跑 4 组对照并输出 mIoU / Precision / Recall / F1 / FPS 表格：

  A. 原始 SAM2        —— 全图自动分割（无提示、无后处理），baseline
  B. SAM2 + 后处理     —— 全图自动分割 + 自研后处理（隔离后处理增益）
  C. YOLO→SAM2 级联    —— ROI 裁剪 + box 提示分割，无后处理（隔离级联增益）
  D. 完整方案          —— 级联 + 后处理（完整系统）

这本质是一个 2×2 因子设计：
  - 提示方式：box（YOLO 级联，C/D） vs 无提示（全图自动，A/B）
  - 后处理：有（B/D） vs 无（A/C）

诚实的设计说明：
  - SAM2 为零样本模型，本项目不做 SAM2 微调；"数据集增强"的收益体现在
    训练 YOLO 检测器阶段（增强训练 → 检测召回率提升），它影响 C/D 组的
    输入质量，而非单独一组。故"仅增强的 SAM2"落实为 B 组（隔离后处理）
    与 C 组（隔离级联）的对照，结论更清晰可复现。
  - A/B 组（无提示）是类无关分割：SAM2 不知道"什么是缺陷"，只输出一批
    区域掩码。故对 A/B/C/D 四组统一用【二值缺陷 IoU】比较（前景=任一缺陷），
    另对 C/D 组额外给【逐类 mIoU】（它们才有类别标签）。A/B 组低 Precision
    正是"无定位、无类别"的诚实体现，也是级联（YOLO 提供定位+类别）价值所在。

运行前提：已完成 YOLO 训练（或至少存在 cfg.detect.model 指向的权重）、
          SAM2 权重就位、测试集与 GT 掩码（data/gt/masks/）已生成。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config as cfg_mod
from src import detect, evaluate, pipeline, postprocess, segment


# --------------------------------------------------------------------------- #
# 分支实现
# --------------------------------------------------------------------------- #
def _size_filtered_union(masks: list[dict], h: int, w: int,
                         min_ratio: float, max_ratio: float) -> np.ndarray:
    """把 SAM2 自动分割结果按面积过滤后取并集（类无关、不泄露 GT）。

    SAM2 无提示会输出成百上千个区域掩码；这里仅按面积占全图比例做类无关
    过滤（阈值与后处理 min/max_area_ratio 一致），再取并集作为"缺陷候选区"。
    """
    out = np.zeros((h, w), dtype=np.uint8)
    total = h * w
    for m in masks:
        seg = m["segmentation"]
        area = float(m.get("area", seg.sum()))
        ratio = area / total
        if min_ratio <= ratio <= max_ratio:
            out[seg] = 1
    return out


def _auto_pred(masks: list[dict], h: int, w: int, cfg,
               apply_post: bool) -> np.ndarray:
    """由 SAM2 自动分割结果生成预测掩码。apply_post 控制是否加自研后处理清洗。

    注意：不调用 postprocess.postprocess_mask 里的全图面积过滤（max_area_ratio），
    因为全图"候选区并集"几乎必然 >90% 会被该阈值误清零，而该面积阈值语义是
    "单个 ROI 缺陷掩码"。这里只施加对全图掩码有意义的清洗：
    孔洞填充 + 连通域碎片剔除 + 自适应形态学平滑。
    """
    pp = cfg.postprocess
    pred = _size_filtered_union(masks, h, w,
                                pp.get("min_area_ratio", 0.0002),
                                pp.get("max_area_ratio", 0.90))
    if apply_post and pred.sum() > 0:
        if pp.get("fill_holes", True):
            pred = postprocess.fill_holes(pred)
        pred = postprocess.remove_fragments(
            pred, pp.get("min_component_ratio", 0.10), keep_largest_only=False)
        pred = postprocess.smooth_contours(
            pred, pp.get("morph_ksize", 5), pp.get("morph_iter", 1))
    return pred


def run_ab_sam2_full(image: np.ndarray, auto_gen, cfg) -> tuple[np.ndarray, np.ndarray, float]:
    """A/B：SAM2 全图自动分割。生成一次，A=原始 B=后处理，共享生成耗时。

    SAM2 全图自动分割是 A/B 组最慢环节，A/B 各调一次会翻倍；这里只生成一次
    掩码，A（apply_post=False）与 B（apply_post=True）复用。
    """
    t0 = time.perf_counter()
    masks = auto_gen.generate(image)
    h, w = image.shape[:2]
    pred_a = _auto_pred(masks, h, w, cfg, apply_post=False)
    pred_b = _auto_pred(masks, h, w, cfg, apply_post=True)
    elapsed = time.perf_counter() - t0
    return pred_a, pred_b, elapsed


def run_c_cascade(image: np.ndarray, yolo, sam2, cfg,
                  use_post: bool = False) -> tuple[np.ndarray, float]:
    """C/D：YOLO→SAM2 级联。use_post 控制是否加自研后处理。

    返回全图类别掩码（0=背景，1..K=类别）。
    """
    t0 = time.perf_counter()
    res = pipeline.run_pipeline(image, cfg, yolo_model=yolo,
                                sam2_predictor=sam2, use_post=use_post)
    elapsed = time.perf_counter() - t0
    return res["mask"], elapsed


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #
def _agg_metrics(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray,
                 classes: list[int]) -> dict:
    p = evaluate.precision(tp, fp)
    r = evaluate.recall(tp, fn)
    f = evaluate.f1_score(p, r)
    iou = evaluate.iou_per_class(tp, fp, fn)
    return {
        "miou": float(np.mean([iou[c] for c in classes])),
        "precision": float(np.mean([p[c] for c in classes])),
        "recall": float(np.mean([r[c] for c in classes])),
        "f1": float(np.mean([f[c] for c in classes])),
    }


def _load_gt_mask(path: Path) -> np.ndarray | None:
    """读 GT 类别掩码 PNG，返回 uint8（0=背景，1..3=类别）。"""
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    return np.asarray(m, dtype=np.uint8)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="消融对比实验（A/B/C/D 四组）")
    ap.add_argument("--test_images", default="data/splits/images/test")
    ap.add_argument("--test_masks", default="data/gt/masks",
                    help="GT 类别掩码目录（class-id 灰度 PNG，0=背景）")
    ap.add_argument("--num_classes", type=int, default=4,
                    help="含背景的类别数（3 缺陷类 + 背景）")
    ap.add_argument("--out_csv", default="experiments/ablation_results.csv")
    ap.add_argument("--out_perclass", default="experiments/ablation_perclass.csv")
    ap.add_argument("--limit", type=int, default=0,
                    help="只评估前 N 张（0=全部，用于快速冒烟测试）")
    ap.add_argument("--model", default=None,
                    help="覆盖 cfg.detect.model 的 YOLO 权重路径（数据消融对比原集模型用）")
    args = ap.parse_args()

    cfg = cfg_mod.get_config()
    if args.model:
        cfg.detect.model = args.model
    class_names = cfg.classes.names  # ["peeling","crack","rust"]

    # 加载模型（级联组 C/D 需要 YOLO；若未训练则降级只跑 A/B）
    auto_gen = segment.build_auto_generator(cfg)
    sam2 = segment.build_sam2(cfg)
    yolo = None
    try:
        yolo = detect.load_yolo(cfg)
        print(f"[消融] 已加载 YOLO 权重：{cfg.detect.model}")
    except Exception as e:  # noqa: BLE001
        print(f"[消融] 警告：加载 YOLO 失败（{e}），C/D 组将跳过，仅跑 A/B。")

    # 收集 GT 掩码 ↔ 测试图对应关系
    mask_dir = Path(args.test_masks)
    img_dir = Path(args.test_images)
    mask_files = sorted(mask_dir.glob("*.png"))
    if args.limit > 0:
        mask_files = mask_files[: args.limit]
    pairs: list[tuple[Path, Path]] = []
    for mf in mask_files:
        img = img_dir / f"{mf.stem}.jpg"
        if img.exists():
            pairs.append((img, mf))
    if not pairs:
        print(f"[消融] 未找到匹配的 GT/测试图（mask_dir={mask_dir} img_dir={img_dir}）")
        return
    print(f"[消融] 评估 {len(pairs)} 张（GT 掩码 {len(mask_files)} 张）")

    nc = args.num_classes
    # 二值缺陷混淆累计：A/B/C/D 四组
    bin_names = ["A_原始SAM2", "B_SAM2+后处理", "C_级联", "D_完整"]
    bin_tp = {n: np.zeros(2, np.int64) for n in bin_names}
    bin_fp = {n: np.zeros(2, np.int64) for n in bin_names}
    bin_fn = {n: np.zeros(2, np.int64) for n in bin_names}
    # 逐类混淆累计：仅 C/D
    cls_tp = {"C_级联": np.zeros(nc, np.int64), "D_完整": np.zeros(nc, np.int64)}
    cls_fp = {"C_级联": np.zeros(nc, np.int64), "D_完整": np.zeros(nc, np.int64)}
    cls_fn = {"C_级联": np.zeros(nc, np.int64), "D_完整": np.zeros(nc, np.int64)}
    # 计时累计
    times = {n: 0.0 for n in bin_names}

    for idx, (img_path, mask_path) in enumerate(pairs, 1):
        image = cv2.imread(str(img_path))
        gt = _load_gt_mask(mask_path)
        if image is None or gt is None:
            continue
        h, w = image.shape[:2]
        gt_bin = (gt > 0).astype(np.uint8)

        # A / B（无提示，共享一次全图自动分割）
        pred_a, pred_b, t_ab = run_ab_sam2_full(image, auto_gen, cfg)
        times["A_原始SAM2"] += t_ab
        times["B_SAM2+后处理"] += t_ab
        for n, pred in [("A_原始SAM2", pred_a), ("B_SAM2+后处理", pred_b)]:
            tp, fp, fn = evaluate.pixel_confusion((pred > 0).astype(np.uint8), gt_bin, 2)
            bin_tp[n] += tp; bin_fp[n] += fp; bin_fn[n] += fn

        # C / D（级联）
        if yolo is not None:
            pred_c, t_c = run_c_cascade(image, yolo, sam2, cfg, use_post=False)
            pred_d, t_d = run_c_cascade(image, yolo, sam2, cfg, use_post=True)
            times["C_级联"] += t_c
            times["D_完整"] += t_d
            for n, pred in [("C_级联", pred_c), ("D_完整", pred_d)]:
                tp, fp, fn = evaluate.pixel_confusion((pred > 0).astype(np.uint8), gt_bin, 2)
                bin_tp[n] += tp; bin_fp[n] += fp; bin_fn[n] += fn
                tp2, fp2, fn2 = evaluate.pixel_confusion(pred, gt, nc)
                cls_tp[n] += tp2; cls_fp[n] += fp2; cls_fn[n] += fn2

        if idx % 20 == 0 or idx == len(pairs):
            print(f"[消融] 已评估 {idx}/{len(pairs)}", flush=True)

    # ---- 汇总二值缺陷指标 ----
    rows = []
    for n in bin_names:
        if times[n] == 0.0 and n in ("C_级联", "D_完整") and yolo is None:
            continue
        m = _agg_metrics(bin_tp[n], bin_fp[n], bin_fn[n], classes=[1])
        fps = len(pairs) / times[n] if times[n] > 0 else 0.0
        rows.append({
            "分支": n,
            "mIoU(缺陷)": round(m["miou"], 4),
            "Precision": round(m["precision"], 4),
            "Recall": round(m["recall"], 4),
            "F1": round(m["f1"], 4),
            "FPS": round(fps, 3),
        })
    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    print("\n========== 消融结果（二值缺陷分割） ==========")
    print(df.to_string(index=False))
    print(f"\n[消融] 已写 {args.out_csv}")

    # ---- 逐类指标（仅 C/D） ----
    if yolo is not None:
        pc_rows = []
        for n in ("C_级联", "D_完整"):
            m = _agg_metrics(cls_tp[n], cls_fp[n], cls_fn[n], classes=list(range(1, nc)))
            row = {"分支": n, "mIoU均值": round(m["miou"], 4),
                   "Precision均值": round(m["precision"], 4),
                   "Recall均值": round(m["recall"], 4), "F1均值": round(m["f1"], 4)}
            iou = evaluate.iou_per_class(cls_tp[n], cls_fp[n], cls_fn[n])
            p = evaluate.precision(cls_tp[n], cls_fp[n])
            r = evaluate.recall(cls_tp[n], cls_fn[n])
            f = evaluate.f1_score(p, r)
            for c in range(1, nc):
                nm = class_names[c - 1] if c - 1 < len(class_names) else f"cls{c}"
                row[f"mIoU_{nm}"] = round(float(iou[c]), 4)
                row[f"F1_{nm}"] = round(float(f[c]), 4)
            pc_rows.append(row)
        df_pc = pd.DataFrame(pc_rows)
        df_pc.to_csv(args.out_perclass, index=False, encoding="utf-8-sig")
        print("\n========== 逐类指标（C/D 级联组） ==========")
        print(df_pc.to_string(index=False))
        print(f"[消融] 已写 {args.out_perclass}")


if __name__ == "__main__":
    main()
