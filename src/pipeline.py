"""完整级联推理链：YOLOv11 检测 → ROI 裁剪 → SAM2 分割 → 后处理 → 量化。

这是单张图片端到端跑通的核心入口，供 Streamlit 与消融实验复用。
"""
from __future__ import annotations

import numpy as np

from . import detect as det
from . import segment as seg
from . import postprocess as pp
from . import quantify as q


def run_pipeline(image_bgr: np.ndarray, cfg,
                 yolo_model=None, sam2_predictor=None,
                 use_post: bool = True) -> dict:
    """单图端到端推理。

    use_post=False 时跳过自研后处理（供消融实验 C 组隔离"级联 vs 后处理"增益）。

    返回 dict：
      results: list[dict]，每个元素含 class/conf/box/mask/pixel_area/ratio/area_mm2
      summary: quantify.summarize 的输出
      mask:   叠加后的全图掩码（HxW，值为 class_id+1，0 为背景）
    """
    if yolo_model is None:
        yolo_model = det.load_yolo(cfg)
    if sam2_predictor is None:
        sam2_predictor = seg.build_sam2(cfg)

    h, w = image_bgr.shape[:2]
    class_names = cfg.classes.names
    mm_per_pixel = cfg.quantify.get("mm_per_pixel", None)

    dets = det.detect(image_bgr, yolo_model, cfg)
    results: list[dict] = []
    full_mask = np.zeros((h, w), dtype=np.uint8)  # 0=背景，>=1 为类别

    for d in dets:
        roi, offset = det.crop_roi(image_bgr, d.xyxy, margin_ratio=0.1)
        if roi.size == 0:
            continue
        # 框坐标从原图转到 ROI 坐标系
        x1, y1, x2, y2 = [int(v) for v in d.xyxy]
        roi_box = np.array([x1 - offset[0], y1 - offset[1],
                            x2 - offset[0], y2 - offset[1]], dtype=np.float32)
        roi_mask = seg.segment_roi(sam2_predictor, roi, roi_box)
        # 自研后处理（创新点3）
        if use_post:
            roi_mask = pp.postprocess_mask(roi_mask, cfg)
        if roi_mask.sum() == 0:
            continue
        placed = det.place_roi_mask(roi_mask, offset, (h, w))

        # 全图掩码叠加：后处理已 keep_largest_only，不同类别不重叠时直接赋值
        full_mask[placed > 0] = int(d.cls) + 1

        quant = q.quantify_mask(placed, d.cls, class_names, h, w, mm_per_pixel)
        results.append({
            "class_id": int(d.cls),
            "class": class_names[d.cls] if d.cls < len(class_names) else f"cls{d.cls}",
            "conf": d.conf,
            "box": [int(v) for v in d.xyxy],
            "mask": placed,
            "pixel_area": quant["pixel_area"],
            "ratio": quant["ratio"],
            "area_mm2": quant["area_mm2"],
        })

    summary = q.summarize(results, h, w, mm_per_pixel)
    return {"results": results, "summary": summary, "mask": full_mask}


def overlay(image_bgr: np.ndarray, mask: np.ndarray, cfg,
            alpha: float = 0.5) -> np.ndarray:
    """生成分割掩码叠加可视化图。mask 值为 class_id+1。"""
    vis = image_bgr.copy()
    colors = cfg.classes.colors
    for cid in range(1, len(colors) + 1):
        cmask = mask == cid
        if cmask.any():
            color = colors[cid - 1]
            vis[cmask] = (vis[cmask] * (1 - alpha) +
                          np.array(color, dtype=np.uint8) * alpha).astype(np.uint8)
    return vis
