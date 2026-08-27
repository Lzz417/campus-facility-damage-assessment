"""生成演示素材：三类缺陷的 原图|分割叠加 对比图 + 批量 Excel 报表样例。

供录屏/截图/申报材料直接使用，避免在 Streamlit 里逐个手动截图。
输出到 outputs/demo/：
  peeling.jpg / crack.jpg / rust.jpg   —— 原图(左) | 分割叠加(右)
  damage_report_sample.xlsx            —— 批量量化报表样例
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config as cfg_mod
from src import detect, pipeline as pl, segment


def main():
    cfg = cfg_mod.get_config()
    yolo = detect.load_yolo(cfg)
    sam2 = segment.build_sam2(cfg)

    out_dir = ROOT / "outputs" / "demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_dir = ROOT / "data" / "gt" / "masks"
    img_dir = ROOT / "data" / "splits" / "images" / "test"

    # GT 掩码中类别值：1=peeling, 2=crack, 3=rust（与消融脚本一致）
    want = {"peeling": 1, "crack": 2, "rust": 3}

    # 每个类别收集最多 8 个候选（GT 含该类且像素足够），逐个试直到检测出该类
    cands: dict[str, list[Path]] = {k: [] for k in want}
    for mf in sorted(gt_dir.glob("*.png")):
        m = cv2.imread(str(mf), cv2.IMREAD_UNCHANGED)
        if m is None:
            continue
        img = img_dir / f"{mf.stem}.jpg"
        if not img.exists():
            continue
        for name, cid in want.items():
            if len(cands[name]) < 8 and (m == cid).sum() > 200:
                cands[name].append(img)

    all_rows: list[dict] = []
    for name in ("peeling", "crack", "rust"):
        saved = False
        for img_path in cands[name]:
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            res = pl.run_pipeline(image, cfg, yolo_model=yolo, sam2_predictor=sam2)
            # 找到检测出该类的结果
            hits = [r for r in res["results"] if r["class"] == name]
            if not hits:
                continue
            overlay = pl.overlay(image, res["mask"], cfg)
            side = cv2.hconcat([image, overlay])
            cv2.imwrite(str(out_dir / f"{name}.jpg"), side)
            print(f"[素材] {name} <- {img_path.name}（检测到 {len(res['results'])} 处缺陷）")
            for r in res["results"]:
                all_rows.append({
                    "图片": img_path.name,
                    "类别": r["class"],
                    "置信度": round(r["conf"], 4),
                    "像素面积": r["pixel_area"],
                    "损伤占比": round(r["ratio"], 6),
                    "真实面积(mm²)": r["area_mm2"] if r["area_mm2"] is not None else "未标定",
                })
            saved = True
            break
        if not saved:
            print(f"[素材] 警告：{name} 未找到能检测出的样本，请降低 conf 或换图")

    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_excel(out_dir / "damage_report_sample.xlsx", index=False,
                    sheet_name="检测结果")
        print(f"[素材] 已写 Excel 样例（{len(all_rows)} 行）")
    print(f"[素材] 输出目录：{out_dir}")


if __name__ == "__main__":
    main()
