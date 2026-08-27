"""Streamlit Web 系统（创新点4）：校园设施损伤智能分割与量化评估。

运行：
  streamlit run app/app.py

功能：单张/批量上传 → YOLO→SAM2 级联推理 → 掩码叠加可视化 →
      缺陷类别/面积/占比量化 → Excel 报表导出。
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import pipeline as pl
from src import quantify as q
from src.config import get_config

st.set_page_config(page_title="校园设施损伤评估系统", layout="wide")


@st.cache_resource
def load_models():
    # 模型只依赖权重文件，与运行时参数(conf/mm_per_pixel)无关，
    # 因此缓存时不带 cfg 参数，避免每次 Streamlit 重跑都重载 2.4GB 的 SAM2。
    from src import detect, segment
    _cfg = get_config()
    yolo = detect.load_yolo(_cfg)
    sam2 = segment.build_sam2(_cfg)
    return yolo, sam2


def process_one(image_bgr, yolo, sam2, cfg):
    res = pl.run_pipeline(image_bgr, cfg, yolo_model=yolo, sam2_predictor=sam2)
    overlay = pl.overlay(image_bgr, res["mask"], cfg)
    return res, overlay


def main():
    st.title("校园设施损伤智能分割与量化评估系统")
    st.caption("YOLOv11 前置检测 + SAM2 分割 + 自适应掩码后处理")

    cfg = get_config()

    with st.sidebar:
        st.header("参数设置")
        mm_per_pixel = st.text_input("标定系数 (mm/像素)", value="",
                                     help="留空则只输出像素面积与占比")
        conf = st.slider("检测置信度", 0.05, 0.95, float(cfg.detect.get("conf", 0.25)))
        cfg.detect.conf = conf
        cfg.quantify.mm_per_pixel = float(mm_per_pixel) if mm_per_pixel.strip() else None
        st.divider()
        st.markdown("**类别**: 墙面剥落 / 地面裂缝 / 设施锈蚀")

    yolo, sam2 = load_models()

    uploaded = st.file_uploader("上传图片（可多选）", type=["jpg", "jpeg", "png", "bmp"],
                                accept_multiple_files=True)

    all_rows: list[dict] = []
    if uploaded:
        for up in uploaded:
            file_bytes = np.frombuffer(up.read(), np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if image is None:
                st.error(f"无法读取 {up.name}")
                continue
            st.subheader(up.name)
            with st.spinner("推理中..."):
                res, overlay = process_one(image, yolo, sam2, cfg)

            col1, col2 = st.columns(2)
            with col1:
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                         caption="原图", use_container_width=True)
            with col2:
                st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                         caption="分割叠加", use_container_width=True)

            st.markdown("#### 量化结果")
            rows = q.to_table_rows(res["summary"], cfg.classes.names)
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)

            # 汇总到批量报表
            for r in res["results"]:
                all_rows.append({
                    "图片": up.name, "类别": r["class"], "置信度": round(r["conf"], 4),
                    "像素面积": r["pixel_area"], "损伤占比": round(r["ratio"], 6),
                    "真实面积(mm²)": r["area_mm2"] if r["area_mm2"] is not None else "未标定",
                })
            if not res["results"]:
                all_rows.append({"图片": up.name, "类别": "无缺陷", "置信度": "-",
                                 "像素面积": 0, "损伤占比": 0.0, "真实面积(mm²)": "-"})

    if all_rows:
        st.divider()
        st.subheader("批量检测报表导出")
        report_df = pd.DataFrame(all_rows)
        st.dataframe(report_df, use_container_width=True)

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            report_df.to_excel(writer, index=False, sheet_name="检测结果")
        st.download_button("下载 Excel 报表", buf.getvalue(),
                           "damage_report.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


if __name__ == "__main__":
    main()
