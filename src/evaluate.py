"""分割指标评估模块：像素级 mIoU / Precision / Recall / F1 / FPS。

掩码约定：0=背景，1..K=缺陷类别。缺陷类别默认从 1 开始，
背景是否计入 mIoU 由 include_background 控制（默认不计入）。
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np


def pixel_confusion(pred: np.ndarray, gt: np.ndarray,
                    num_classes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """像素级混淆统计，返回每类 (TP, FP, FN)。"""
    pred = np.asarray(pred).astype(np.int64)
    gt = np.asarray(gt).astype(np.int64)
    tp = np.zeros(num_classes, dtype=np.int64)
    fp = np.zeros(num_classes, dtype=np.int64)
    fn = np.zeros(num_classes, dtype=np.int64)
    for c in range(num_classes):
        p = (pred == c)
        g = (gt == c)
        tp[c] = int((p & g).sum())
        fp[c] = int((p & ~g).sum())
        fn[c] = int((~p & g).sum())
    return tp, fp, fn


def iou_per_class(tp, fp, fn) -> np.ndarray:
    denom = tp.astype(np.float64) + fp + fn
    out = np.where(denom > 0, tp.astype(np.float64) / np.where(denom > 0, denom, 1), 0.0)
    return out


def miou(tp, fp, fn, classes: list[int] | None = None) -> float:
    """mIoU。classes 指定参与统计的类别 id（默认 1..K 缺陷类）。"""
    iou = iou_per_class(tp, fp, fn)
    if classes is None:
        classes = list(range(1, len(tp)))
    if not classes:
        return 0.0
    return float(np.mean([iou[c] for c in classes]))


def precision(tp, fp) -> np.ndarray:
    denom = tp.astype(np.float64) + fp
    return np.where(denom > 0, tp.astype(np.float64) / np.where(denom > 0, denom, 1), 0.0)


def recall(tp, fn) -> np.ndarray:
    denom = tp.astype(np.float64) + fn
    return np.where(denom > 0, tp.astype(np.float64) / np.where(denom > 0, denom, 1), 0.0)


def f1_score(p, r) -> np.ndarray:
    denom = p + r
    return np.where(denom > 0, 2 * p * r / np.where(denom > 0, denom, 1), 0.0)


def evaluate_image(pred: np.ndarray, gt: np.ndarray,
                   num_classes: int, classes: list[int] | None = None) -> dict:
    """单图评估，返回 mIoU / mean Precision / mean Recall / mean F1。"""
    tp, fp, fn = pixel_confusion(pred, gt, num_classes)
    if classes is None:
        classes = list(range(1, num_classes))
    p = precision(tp, fp)
    r = recall(tp, fn)
    f = f1_score(p, r)
    return {
        "miou": miou(tp, fp, fn, classes),
        "precision": float(np.mean([p[c] for c in classes])) if classes else 0.0,
        "recall": float(np.mean([r[c] for c in classes])) if classes else 0.0,
        "f1": float(np.mean([f[c] for c in classes])) if classes else 0.0,
    }


def benchmark_fps(run_fn: Callable, warmup: int = 2, runs: int = 10) -> float:
    """测推理帧率（FPS）。run_fn 执行一次完整推理。"""
    for _ in range(warmup):
        run_fn()
    if torch_available():
        import torch
        torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    for _ in range(runs):
        run_fn()
    if torch_available():
        import torch
        torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.perf_counter() - t0
    return runs / elapsed if elapsed > 0 else 0.0


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False
