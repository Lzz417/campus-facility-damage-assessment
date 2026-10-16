"""完整级联推理链：YOLOv11 检测 → ROI 裁剪 → SAM2 分割 → 后处理 → 量化。

这是单张图片端到端跑通的核心入口，供 Streamlit 与消融实验复用。
"""
from __future__ import annotations

import cv2
import numpy as np

from . import detect as det
from . import segment as seg
from . import postprocess as pp
from . import quantify as q


def _adjust_box(roi_box: np.ndarray, cfg, cls_name: str | None,
                roi_shape: tuple[int, int]) -> np.ndarray:
    """按类别对 SAM2 box prompt 做边距调整（配置可调，正=扩张、负=收缩）。

    默认关闭（config.yaml 中 box_margin 全 0）。实测定论（198 张 GT 全量）：
      * 面病害（剥落/锈蚀）扩张让 SAM2 把周边墙面/地面误当缺陷，掩码过分割，
        剥落 mIoU 0.757→0.232、锈蚀 0.688→0.512，整体缺陷 mIoU 0.692→0.331。
      * 线状裂缝负边距（收缩）会使部分裂缝掩码膨胀 2.5 倍（背景外溢）。
    保留此函数作为实验调参选项；开启时扩张引入的背景外溢由后处理清洗
    （keep_largest_only + min_area_px）。
    """
    margins = cfg.detect.get("box_margin") or {}
    ratio = 0.0
    if hasattr(margins, "get"):
        ratio = float(margins.get(cls_name, 0.0) or 0.0)
    if ratio == 0.0:
        return roi_box
    bw = roi_box[2] - roi_box[0]
    bh = roi_box[3] - roi_box[1]
    if bw <= 0 or bh <= 0:
        return roi_box
    box = roi_box.copy()
    box[0] -= bw * ratio
    box[1] -= bh * ratio
    box[2] += bw * ratio
    box[3] += bh * ratio
    rh, rw = roi_shape[:2]
    # 边界钳制：坐标不能 <0、不能超出 ROI 宽高
    box[[0, 2]] = np.clip(box[[0, 2]], 0, rw - 1)
    box[[1, 3]] = np.clip(box[[1, 3]], 0, rh - 1)
    if box[2] <= box[0] or box[3] <= box[1]:  # 防退化
        return roi_box
    return box


def run_pipeline(image_bgr: np.ndarray, cfg,
                 yolo_model=None, sam2_predictor=None,
                 use_post: bool = True,
                 frozen_compat: bool = False) -> dict:
    """单图端到端推理。

    use_post=False 时跳过自研后处理（供消融实验 C 组隔离"级联 vs 后处理"增益）。

    frozen_compat=False（**默认，保持历史行为不变**）：
        * ROI 以 BGR 原样送 SAM2；
        * 叠加按 conf 降序遍历 ⇒ 跨类冲突时**低 conf 胜**。
        历史调用方（`experiments/ablation.py`、`scripts/eval_box_margin_effect.py`、
        `scripts/eval_branch_d.py`、`scripts/make_demo_assets.py`）依赖这个行为，
        改它会让它们无法复现已记录的 `experiments/ablation*.csv`。

    frozen_compat=True（**仅 Web demo 使用**）：
        * 送 SAM2 前显式 BGR→RGB；
        * 叠加按 conf 升序 ⇒ 跨类冲突时**高 conf 胜**。
        两者都与 V2 冻结评价脚本 `eval_seg_cd_test.py` 的口径一致。

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

    # 遍历顺序：frozen_compat=True 时按 conf **升序**（低 conf 先写、高 conf 后写）
    # ⇒ 跨类冲突时**高 conf 胜**，与冻结评价口径一致（eval_seg_cd_test.py 同样升序放置）。
    # 默认（False）保持 det.detect 的降序 ⇒ 后写的低 conf 覆盖 = **低 conf 胜**（历史行为）。
    # 注意：只对**遍历副本**排序，绝不修改 det.detect 本身的降序语义。
    iter_dets = sorted(dets, key=lambda x: x.conf) if frozen_compat else dets
    for d in iter_dets:
        roi, offset = det.crop_roi(image_bgr, d.xyxy, margin_ratio=0.1)
        if roi.size == 0:
            continue
        # 框坐标从原图转到 ROI 坐标系
        x1, y1, x2, y2 = [int(v) for v in d.xyxy]
        roi_box = np.array([x1 - offset[0], y1 - offset[1],
                            x2 - offset[0], y2 - offset[1]], dtype=np.float32)
        # 逐类 box 边距调整（默认关闭，见 _adjust_box 注释；实验项）
        cls_name = class_names[d.cls] if d.cls < len(class_names) else None
        roi_box = _adjust_box(roi_box, cfg, cls_name, roi.shape)
        # frozen_compat=True 时显式 BGR→RGB 后再送 SAM2（SAM2 期望 RGB 输入）。
        # 只在**这一侧**转换——src/segment.py 不做转换，因为冻结评价脚本
        # （eval_seg_cd_test.py）已经自己传 RGB，若在那里再转一次会造成双重转换、
        # 破坏 V2 C/D 掩码的可复现性。
        roi_for_sam = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB) if frozen_compat else roi
        roi_mask = seg.segment_roi(sam2_predictor, roi_for_sam, roi_box)
        # 自研后处理（创新点3）
        if use_post:
            min_px_cfg = cfg.postprocess.get("min_area_px_per_class") or {}
            min_px = int(min_px_cfg.get(cls_name, 0) or 0) if hasattr(min_px_cfg, "get") else 0
            roi_mask = pp.postprocess_mask(roi_mask, cfg, min_area_px=min_px)
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
