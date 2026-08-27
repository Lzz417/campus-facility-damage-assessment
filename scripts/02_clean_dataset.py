"""数据集清洗 + 类别重构（创新点1 前半）。

将公开数据集的原始标注统一映射为系统的 3 个缺陷类：
  0=peeling 墙面剥落  1=crack 地面裂缝  2=rust 设施锈蚀
（"完好" 作为负样本单独保留，不设检测类别。）

支持两种原始标注：
  1) MBDD2025 —— Pascal VOC XML（只有 bbox，无掩码）
  2) CrackForest —— 掩码图（像素级）

说明：MBDD 只提供检测框，因此这里仅做"筛选 + 重分类"生成 YOLO 检测标签；
分割掩码将由 SAM2 在推理阶段生成（伪标签），评估所需的 GT 掩码用
CrackForest + 人工复核子集构建。
"""
from __future__ import annotations

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import yaml

# 原始类别 → 系统类别 映射（其余类别剔除）
MBDD_CLASS_MAP = {
    "crack": 1,        # 裂缝 → 地面裂缝
    "abscission": 0,   # 剥落 → 墙面剥落
    "corrosion": 2,    # 锈蚀 → 设施锈蚀
    # leakage(渗漏)、bulge(鼓包) 等不映射 → 剔除
}
NEGATIVE_CLASSES = {"intact", "normal", "good"}  # 完好样本标记


def parse_voc(xml_path: Path) -> list[dict]:
    """解析 VOC XML，返回 [{'class': str, 'xyxy': (x1,y1,x2,y2)}]。"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objs = []
    for obj in root.iter("object"):
        name = obj.findtext("name", "").strip().lower()
        bbox = obj.find("bndbox")
        if bbox is None:
            continue
        xyxy = (int(float(bbox.findtext("xmin"))), int(float(bbox.findtext("ymin"))),
                int(float(bbox.findtext("xmax"))), int(float(bbox.findtext("ymax"))))
        objs.append({"class": name, "xyxy": xyxy})
    return objs


def to_yolo_line(xyxy: tuple, cls: int, w: int, h: int) -> str:
    x1, y1, x2, y2 = xyxy
    cx = ((x1 + x2) / 2) / w
    cy = ((y1 + y2) / 2) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def main():
    ap = argparse.ArgumentParser(description="数据集清洗 + 类别重构")
    ap.add_argument("--src_images", default="data/raw/MBDD/MBDD2025/JPEGImages")
    ap.add_argument("--src_labels", default="data/raw/MBDD/MBDD2025/Annotations")  # VOC xml 目录
    ap.add_argument("--out_images", default="data/processed/images")
    ap.add_argument("--out_labels", default="data/processed/labels")  # YOLO txt
    ap.add_argument("--out_negative", default="data/processed/negative")
    args = ap.parse_args()

    src_img = Path(args.src_images)
    src_lbl = Path(args.src_labels)
    out_img = Path(args.out_images)
    out_lbl = Path(args.out_labels)
    out_neg = Path(args.out_negative)
    for d in (out_img, out_lbl, out_neg):
        d.mkdir(parents=True, exist_ok=True)

    kept, dropped, negative = 0, 0, 0
    for xml in sorted(src_lbl.glob("*.xml")):
        objs = parse_voc(xml)
        mapped = [o for o in objs if o["class"] in MBDD_CLASS_MAP]
        has_negative = any(o["class"] in NEGATIVE_CLASSES for o in objs)

        stem = xml.stem
        img_file = src_img / f"{stem}.jpg"
        if not img_file.exists():
            img_file = src_img / f"{stem}.png"
        if not img_file.exists():
            dropped += 1
            continue

        # 完好样本（无缺陷标注）→ 负样本
        if not mapped and (has_negative or not objs):
            shutil.copy2(img_file, out_neg / img_file.name)
            negative += 1
            continue

        if not mapped:
            dropped += 1
            continue

        img = cv2.imread(str(img_file))
        h, w = img.shape[:2]
        lines = [to_yolo_line(o["xyxy"], MBDD_CLASS_MAP[o["class"]], w, h)
                 for o in mapped]

        shutil.copy2(img_file, out_img / img_file.name)
        (out_lbl / f"{stem}.txt").write_text("\n".join(lines) + "\n",
                                             encoding="utf-8")
        kept += 1

    print(f"[清洗] 保留 {kept} 张，负样本 {negative} 张，剔除 {dropped} 张")
    print("[清洗] YOLO 标签已写入", out_lbl)


if __name__ == "__main__":
    main()
