"""第二级：SAM2 零样本分割（创新点2 后半）。

对 YOLOv11 裁剪出的 ROI 做提示式分割（box prompt），输出二值掩码。
SAM2 使用官方 facebookresearch/sam2 包，模型权重需预先下载到 weights/。
"""
from __future__ import annotations

import glob
import os
from typing import Optional

import numpy as np


def _resolve_config(cfg) -> str:
    """定位 sam2 包内的模型 config yaml（sam2_hiera_l.yaml 等）。"""
    import sam2
    base = os.path.dirname(sam2.__file__)
    name = cfg.segment.config
    matches = glob.glob(os.path.join(base, "**", name), recursive=True)
    return matches[0] if matches else name


def build_sam2(cfg):
    """构建 SAM2 图像预测器（延迟导入 + 自动回退 CPU）。"""
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = cfg.segment.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    config_path = _resolve_config(cfg)
    sam2_model = build_sam2(config_file=config_path,
                            ckpt_path=cfg.segment.checkpoint,
                            device=device,
                            apply_postprocessing=False)  # 后处理交给自研模块
    return SAM2ImagePredictor(sam2_model)


def build_auto_generator(cfg):
    """构建 SAM2 全图自动分割器（无 box/point 提示，消融实验 A/B 组用）。"""
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    predictor = build_sam2(cfg)
    # SAM2ImagePredictor.model 即底层 SAM2 模型，直接交给自动分割器复用
    return SAM2AutomaticMaskGenerator(model=predictor.model)


def segment_roi(predictor, roi_bgr: np.ndarray,
                box_xyxy: np.ndarray) -> np.ndarray:
    """对 ROI 做 box 提示分割，返回 0/1 二值掩码。

    box_xyxy 为 ROI 内的像素坐标（相对 ROI 左上角）。
    """
    predictor.set_image(roi_bgr)
    # SAM2 期望 box 为 (1, 4) 或 (B, 4)，这里统一 reshape 避免维度错误
    box = np.asarray(box_xyxy, dtype=np.float32).reshape(1, 4)
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=box,
        multimask_output=False,
    )
    # SAM2 返回 (1, H, W)（multimask_output=False 时只有 1 个掩码），
    # 取 [0] 得到 (H, W) 二维掩码，避免下游 _binary 误把 (1,H,W) 当 3 通道
    # 处理而只取到最左一列（曾导致裂缝掩码被清成 3px）。
    mask = np.asarray(masks)[0]
    return (mask > 0).astype(np.uint8)
