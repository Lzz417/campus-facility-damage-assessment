"""自适应掩码后处理模块（创新点3）。

针对 SAM2 原生输出存在的三类问题做修复：
1) 伪掩码 / 过分割误检   —— 面积阈值过滤 + 连通域碎片剔除
2) 小目标、遮挡缺陷轮廓断裂 —— 孔洞填充 + 形态学闭运算
3) 边缘毛刺 / 锯齿       —— 自适应形态学平滑（细长裂缝仅闭运算，斑块状才开运算）

全部为纯 numpy / opencv / scipy 实现，无需 GPU，可独立单元测试。
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage


def _binary(mask: np.ndarray) -> np.ndarray:
    """归一化为 0/1 二值 uint8 掩码。"""
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return (mask > 0).astype(np.uint8)


def _is_thin(mask: np.ndarray, ksize: int = 5, keep_ratio: float = 0.30) -> bool:
    """腐蚀探针判形：腐蚀后前景占比 < keep_ratio 说明以细长/纤细结构为主（如裂缝）。

    5×5 椭圆核 OPEN（先腐蚀）会从每边削掉约 2px：宽度 w 的线腐蚀后保留约
    (w-4)/w（w>4），w=5→20%、w=6→33%。故阈值取 0.30 可干净区分"细长线
    (OPEN 会损伤)"与"实心斑块(OPEN 安全)"。细长结构应跳过 OPEN，且不应
    只保留最大连通域（裂缝天然断裂成多段）。
    """
    mask = _binary(mask)
    if mask.sum() == 0:
        return False
    ksize = max(3, ksize if ksize % 2 == 1 else ksize + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    eroded = cv2.erode(mask, kernel)
    return float(eroded.sum()) / float(mask.sum()) < keep_ratio


def filter_by_area(mask: np.ndarray, min_ratio: float = 0.0002,
                   max_ratio: float = 0.90) -> np.ndarray:
    """面积阈值过滤：剔除过小伪掩码与过大整图误分割。

    面积占比 = 前景像素 / 全图像素。返回 0/1 掩码（若被剔除则全 0）。
    """
    mask = _binary(mask)
    h, w = mask.shape[:2]
    total = h * w
    if total == 0:
        return mask
    ratio = float(mask.sum()) / total
    if ratio < min_ratio or ratio > max_ratio:
        return np.zeros_like(mask)
    return mask


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """孔洞填充：将前景内部被包围的空洞填为前景，修复轮廓断裂中空。"""
    mask = _binary(mask)
    return ndimage.binary_fill_holes(mask).astype(np.uint8)


def remove_fragments(mask: np.ndarray, min_component_ratio: float = 0.10,
                     keep_largest_only: bool = True) -> np.ndarray:
    """连通域分析去碎片。

    - keep_largest_only=True：只保留面积最大的连通域；
    - 否则：保留面积 >= 最大域 * min_component_ratio 的连通域。
    """
    mask = _binary(mask)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:  # 无前景
        return mask
    # 跳过 label 0（背景）
    areas = stats[1:, cv2.CC_STAT_AREA]
    if areas.size == 0:
        return mask
    largest_idx = int(np.argmax(areas))
    largest_area = int(areas[largest_idx])
    out = np.zeros_like(mask)
    for i, area in enumerate(areas):
        if keep_largest_only:
            keep = (i == largest_idx)
        else:
            keep = area >= largest_area * min_component_ratio
        if keep:
            out[labels == (i + 1)] = 1
    return out


def smooth_contours(mask: np.ndarray, ksize: int = 5,
                    iterations: int = 1,
                    thin_keep_ratio: float = 0.30) -> np.ndarray:
    """自适应形态学平滑：先闭填补断裂，再据结构形状决定是否开运算。

    细长缺陷（如裂缝）宽仅数像素，直接做 OPEN（先腐蚀）会被整条腐蚀消失。
    用腐蚀探针 _is_thin 预判：细长结构只做 CLOSE 连接断裂、跳过 OPEN；
    斑块状缺陷（剥落/锈蚀）才做 CLOSE+OPEN 去毛刺。阈值推导见 _is_thin。
    """
    mask = _binary(mask)
    ksize = max(3, ksize if ksize % 2 == 1 else ksize + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    thin = _is_thin(mask, ksize, thin_keep_ratio)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    if thin:
        return closed  # 细长结构：仅闭运算连接断裂，避免 OPEN 腐蚀消失
    return cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=iterations)


def postprocess_mask(mask: np.ndarray, cfg, min_area_px: int = 0) -> np.ndarray:
    """完整后处理流水线。cfg 为 config.py 的 Config 对象（.postprocess 节）。

    min_area_px：按类别传入的最小掩码绝对像素数（0 表示不启用），
    用于剔除低置信虚检产生的极小无效掩码（不渲染、不进量化表）。

    输入任意形状掩码，输出 0/1 uint8 二值掩码。
    """
    pp = cfg.postprocess
    mask = _binary(mask)
    if mask.sum() == 0:
        return mask
    mask = filter_by_area(mask, pp.get("min_area_ratio", 0.0002),
                          pp.get("max_area_ratio", 0.90))
    if mask.sum() == 0:
        return mask
    if min_area_px > 0 and int(mask.sum()) < min_area_px:
        return np.zeros_like(mask)
    if pp.get("fill_holes", True):
        mask = fill_holes(mask)
    # 自适应：细长缺陷（裂缝）断裂成多段，不应只保留最大连通域；
    # 斑块状缺陷（剥落/锈蚀）则保留最大域、剔除散落碎片。
    thin = _is_thin(mask, pp.get("morph_ksize", 5))
    mask = remove_fragments(mask,
                            pp.get("min_component_ratio", 0.10),
                            keep_largest_only=(not thin and pp.get("keep_largest_only", True)))
    mask = smooth_contours(mask,
                           pp.get("morph_ksize", 5),
                           pp.get("morph_iter", 1))
    return _binary(mask)


if __name__ == "__main__":
    # 简单自测：构造一个带孔洞 + 碎片 + 噪声的掩码
    import sys
    sys.path.insert(0, ".")
    from src.config import get_config

    test = np.zeros((200, 200), dtype=np.uint8)
    test[50:150, 50:150] = 1           # 主缺陷
    test[60:140, 60:140] = 0           # 内部孔洞
    test[5:15, 5:15] = 1               # 碎片
    test[180:185, 180:195] = 1         # 碎片2
    rng = np.random.default_rng(0)
    test[test == 1] = 1                # 保持不变

    cfg = get_config()
    out = postprocess_mask(test, cfg)
    print("原始前景像素:", int(test.sum()))
    print("后处理前景像素:", int(out.sum()))
    print("后处理是否已填孔洞(中心点应为1):", out[100, 100])
