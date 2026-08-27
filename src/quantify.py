"""损伤量化模块（创新点4 的数据部分）。

由分割掩码统计：像素面积、损伤占比、真实面积（需标定系数 mm/像素），
并按类别汇总，供 Streamlit 前端与 Excel 报表直接使用。
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


def pixel_area(mask: np.ndarray) -> int:
    """前景像素数量。"""
    return int((np.asarray(mask) > 0).sum())


def defect_ratio(mask: np.ndarray, h: int, w: int) -> float:
    """损伤占比 = 缺陷像素 / 全图像素。"""
    if h * w == 0:
        return 0.0
    return pixel_area(mask) / (h * w)


def pixels_to_area_mm2(pixels: int, mm_per_pixel: float | None) -> float | None:
    """像素面积换算真实面积（mm²）。未标定时返回 None。"""
    if mm_per_pixel is None:
        return None
    return float(pixels) * float(mm_per_pixel) ** 2


def quantify_mask(mask: np.ndarray, class_id: int, class_names: list[str],
                  h: int, w: int, mm_per_pixel: float | None) -> dict:
    """单类量化结果。"""
    px = pixel_area(mask)
    return {
        "class_id": int(class_id),
        "class": class_names[class_id] if 0 <= class_id < len(class_names) else f"cls{class_id}",
        "pixel_area": px,
        "ratio": defect_ratio(mask, h, w),
        "area_mm2": pixels_to_area_mm2(px, mm_per_pixel),
    }


def summarize(results: Iterable[dict], h: int, w: int,
              mm_per_pixel: float | None) -> dict:
    """汇总：分列各类别 + 总损伤占比。"""
    results = list(results)
    total_px = sum(r["pixel_area"] for r in results)
    summary = {
        "num_defects": len(results),
        "total_pixel_area": total_px,
        "total_ratio": total_px / (h * w) if h * w else 0.0,
        "total_area_mm2": pixels_to_area_mm2(total_px, mm_per_pixel),
        "per_class": results,
    }
    return summary


def to_table_rows(summary: dict, class_names: list[str]) -> list[dict]:
    """把 summary 转成 pandas 友好的行列表（供 Excel 导出）。"""
    rows = []
    for r in summary["per_class"]:
        rows.append({
            "类别": r["class"],
            "像素面积": r["pixel_area"],
            "损伤占比": round(r["ratio"], 6),
            "真实面积(mm²)": round(r["area_mm2"], 3) if r["area_mm2"] is not None else "未标定",
        })
    rows.append({
        "类别": "总计",
        "像素面积": summary["total_pixel_area"],
        "损伤占比": round(summary["total_ratio"], 6),
        "真实面积(mm²)": round(summary["total_area_mm2"], 3) if summary["total_area_mm2"] is not None else "未标定",
    })
    return rows
