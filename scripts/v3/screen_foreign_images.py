"""筛除外来图像（截图 / 实验室试片 / 图库与商品照），并把「无法用文件名判定」的部分
做成**带编号的密排 contact sheet**，供人/agent 逐个目视剔除。

为什么必须两步：
  * 文件名能确定地抓住的：屏幕截图（`Captura-de-Tela*`、`Screenshot*`）、
    涂层实验室试片（`*_steel-*` 且含 `MIL-DTL`/`abrasive-blasted`/`phosphate`/`silane`/
    `chromated`/`mill-finish`）、图库 ID（`photo-<时间戳>-<hex>`）。
  * 文件名**抓不到**的：第三方水印图、商品/影棚照、非设施主体（船、汽车）。
    这些只能看图。所以本脚本把剔除规则**逐条记账**，并把剩余部分**编号出屏**。

用法：
  # 1) 只做确定性剔除，看剩余规模
  ... screen_foreign_images.py --images-manifest <csv> --class corrosion --out <dir>
  # 2) 把剩余部分出屏（30 张/屏，带编号）
  ... screen_foreign_images.py --images-manifest <csv> --class corrosion --out <dir> --sheets
  # 3) 把 agent 目视判定为「可疑」的编号写进 manual_exclusions.txt（一行一个 stem），再跑一次
"""
import argparse
import csv
import pathlib
import re
import sys

import cv2
import numpy as np

# ---- 确定性规则（每条都写清理由，便于复核）----
RULES = [
    ("screenshot",
     r"(Captura[-_ ]?de[-_ ]?Tela|Screenshot|screen[-_ ]?shot|screen[-_ ]?capture)",
     "屏幕截图：不是现场照片"),
    ("lab_coupon",
     r"(_steel[-_]|steel-high-hard|MIL-DTL|abrasive-blasted|zinc-phosphate|silane-pretreatment"
     r"|chromated-wash-primer|mill-finish)",
     "涂层实验室试片：非现场设施"),
    ("stock_id",
     r"(photo-\d{10,}-[0-9a-f]+)",
     "图库照片 ID（Unsplash 风格命名）"),
    ("stock_watermark_name",
     r"(shutterstock|alamy|dreamstime|istock|gettyimages|inspectapedia|123rf|stockphoto)",
     "命名里直接带图库/水印站名字"),
    # --- 以下四条来自对 corrosion 全部 427 张的逐屏目视审计（18 屏）后确认的网络来源特征 ---
    ("ai_generated",
     r"craiyon",
     "AI 生成图：文件名就是 craiyon（AI 图像生成器）的产物"),
    ("news_vendor_article",
     r"(ImageForNews|VUETRADE|wreak-havoc|water-heater)",
     "新闻/厂商文章配图，不是现场采集"),
    ("stock_descriptive_title",
     r"((rusty|rust|corrosion|corrosione|galvanic|pitting|crevice|oxidation)[a-z]*[-_]|"
     r"severe-case-of|Stainless-steel|Rostige|close-up-shot-of|over-white|abstack|abfad|"
     r"-cropped|wheel-arches|rebar-|-bolts-)",
     "图库的「描述式标题」命名（如 rusty-bolts-with-nuts-over-white）"),
    ("stock_asset_id",
     r"(\d{2,}_F_[A-Za-z0-9]{6,}|depositphotos|gettyimages|stockphoto)",
     "图库素材编号（如 1000_F_80078307…/340_F_43650824… 为图库编号）"),
]


def fam_clean(stem):
    s = re.sub(r"\.(jpg|jpeg|png|jfif)\.rf\.[0-9a-f]+$", "", stem, flags=re.I)
    s = re.sub(r"\.rf\.[0-9a-f]+$", "", s, flags=re.I)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-manifest", required=True)
    ap.add_argument("--dataset-root", default=None, help="出屏时必需")
    ap.add_argument("--class", dest="cls", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sheets", action="store_true")
    ap.add_argument("--per-sheet", type=int, default=30)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--cell-w", type=int, default=300)
    ap.add_argument("--cell-h", type=int, default=220)
    ap.add_argument("--only", default=None,
                    help="只出屏这些 stem（逗号分隔）；缺省则出屏全部剩余")
    ap.add_argument("--thumb-limit", type=int, default=0, help=">0 时只出前 N 张（抽查用）")
    a = ap.parse_args()

    out = pathlib.Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(a.images_manifest, encoding="utf-8")))
    sel = [r for r in rows if a.cls in r["classes_present_names"].split("|")]

    manual = set()
    mf = out / "manual_exclusions.txt"
    if mf.exists():
        manual = {l.strip() for l in mf.read_text(encoding="utf-8").splitlines()
                  if l.strip() and not l.startswith("#")}

    excl, kept = [], []
    for r in sel:
        name = fam_clean(r["stem"])
        hit = None
        for rid, pat, why in RULES:
            if re.search(pat, name, re.I):
                hit = (rid, why)
                break
        if hit:
            excl.append({"stem": r["stem"], "split": r["split"], "rule": hit[0],
                         "reason": hit[1], "source": "filename_rule"})
        elif r["stem"] in manual:
            excl.append({"stem": r["stem"], "split": r["split"], "rule": "manual_visual",
                         "reason": "目视判定为外来/可疑", "source": "manual"})
        else:
            kept.append(r)

    with open(out / ("excluded_%s.csv" % a.cls), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["stem", "split", "rule", "reason", "source"])
        w.writeheader()
        w.writerows(excl)
    with open(out / ("remaining_%s.txt" % a.cls), "w", encoding="utf-8") as f:
        for r in kept:
            f.write("%s\t%s\t%s\n" % (r["split"], r["stem"], r["rel_path"]))

    by_rule = {}
    for e in excl:
        by_rule[e["rule"]] = by_rule.get(e["rule"], 0) + 1
    print("[%s] 类内共 %d 张 → 剔除 %d 张 %s，剩余 %d 张"
          % (a.cls, len(sel), len(excl), by_rule, len(kept)))

    if a.sheets:
        if not a.dataset_root:
            print("[FATAL] --sheets 需要 --dataset-root")
            return 2
        root = pathlib.Path(a.dataset_root).resolve()
        pool = kept
        if a.only:
            want = {s.strip() for s in a.only.split(",") if s.strip()}
            pool = [r for r in pool if r["stem"] in want]
        if a.thumb_limit:
            pool = pool[:a.thumb_limit]
        per = a.per_sheet
        rows_n = (per + a.cols - 1) // a.cols
        cw, ch = a.cell_w, a.cell_h
        n_sheet = 0
        for s in range((len(pool) + per - 1) // per):
            chunk = pool[s * per:(s + 1) * per]
            sheet = np.full((rows_n * ch + 30, a.cols * cw, 3), 28, np.uint8)
            cv2.putText(sheet, "%s visual screen | %d of %d remaining | sheet %d/%d "
                              "| number = index in remaining_%s.txt"
                        % (a.cls, len(chunk), len(pool), s + 1,
                           (len(pool) + per - 1) // per, a.cls),
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            for k, r in enumerate(chunk):
                rr, cc = divmod(k, a.cols)
                y, x = 30 + rr * ch, cc * cw
                im = cv2.imread(str(root / r["rel_path"]), cv2.IMREAD_COLOR)
                if im is not None:
                    h, w = im.shape[:2]
                    sc = min((cw - 4) / w, (ch - 18) / h)
                    rs = cv2.resize(im, (max(1, int(w * sc)), max(1, int(h * sc))))
                    sheet[y:y + rs.shape[0], x:x + rs.shape[1]] = rs
                gi = s * per + k
                cv2.rectangle(sheet, (x, y), (x + 44, y + 22), (0, 0, 0), -1)
                cv2.putText(sheet, "#%d" % gi, (x + 3, y + 17),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
                nm = re.sub(r"[^\x20-\x7e]", "?", r["stem"])[:34]
                cv2.putText(sheet, nm, (x + 4, y + ch - 4), cv2.FONT_HERSHEY_SIMPLEX,
                            0.34, (190, 190, 190), 1, cv2.LINE_AA)
            cv2.imwrite(str(out / ("screen_%s_%02d.jpg" % (a.cls, s + 1))), sheet,
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
            n_sheet += 1
        print("[%s] 出屏 %d 张（%d 屏）→ %s" % (a.cls, len(pool), n_sheet, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
