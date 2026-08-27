"""第一级：YOLOv11 缺陷前置检测 + ROI 裁剪（创新点2 前半）。

用 YOLOv11 快速定位缺陷 bounding box，再裁剪 ROI 送入 SAM2，
避免对整张大图做全图分割，缩减无效计算量。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class Detection:
    """一个检测框。xyxy 为 (x1, y1, x2, y2) 归一化或像素坐标。"""
    xyxy: np.ndarray
    conf: float
    cls: int


def load_yolo(cfg):
    """加载 YOLOv11 模型（延迟导入，避免未装依赖时 import 报错）。"""
    from ultralytics import YOLO
    return YOLO(cfg.detect.model)


def detect(image_bgr: np.ndarray, model, cfg) -> list[Detection]:
    """对整图做目标检测，返回缺陷框列表（按置信度降序）。"""
    from ultralytics import YOLO  # noqa: F401 确保依赖已导入

    results = model.predict(
        source=image_bgr,
        conf=cfg.detect.get("conf", 0.25),
        iou=cfg.detect.get("iou", 0.45),
        imgsz=cfg.detect.get("imgsz", 640),
        max_det=cfg.detect.get("max_det", 30),
        verbose=False,
    )
    dets: list[Detection] = []
    if not results or results[0].boxes is None:
        return dets
    boxes = results[0].boxes
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)
    for b, c, cl in zip(xyxy, confs, clss):
        dets.append(Detection(xyxy=b, conf=float(c), cls=int(cl)))
    dets.sort(key=lambda d: d.conf, reverse=True)
    return dets


def crop_roi(image_bgr: np.ndarray, xyxy: np.ndarray,
             margin_ratio: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """按框裁剪 ROI 并外扩 margin，返回 (roi, offset_xy)。

    offset_xy 为 ROI 左上角在原图的 (x0, y0)，用于把 ROI 掩码映射回原图。
    """
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    mw = int((x2 - x1) * margin_ratio)
    mh = int((y2 - y1) * margin_ratio)
    x1 = max(0, x1 - mw)
    y1 = max(0, y1 - mh)
    x2 = min(w, x2 + mw)
    y2 = min(h, y2 + mh)
    roi = image_bgr[y1:y2, x1:x2]
    return roi, np.array([x1, y1], dtype=int)


def place_roi_mask(roi_mask: np.ndarray, offset: np.ndarray,
                   full_shape: tuple[int, int]) -> np.ndarray:
    """把 ROI 掩码按 offset 放回原图尺寸的空白掩码上。"""
    h, w = full_shape
    out = np.zeros((h, w), dtype=np.uint8)
    rh, rw = roi_mask.shape[:2]
    x0, y0 = int(offset[0]), int(offset[1])
    out[y0:y0 + rh, x0:x0 + rw] = (roi_mask > 0).astype(np.uint8)
    return out
