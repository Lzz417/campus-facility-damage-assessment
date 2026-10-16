"""Smart Campus Dataset 完整性审计（只读输入，只写审计产物）。

适配 Roboflow 的 YOLO detection 导出布局：
    <root>/data.yaml
    <root>/{train,valid,test}/images/*.jpg|jpeg|png
    <root>/{train,valid,test}/labels/*.txt      # 每行: class cx cy w h（归一化）

输出（默认写到 --out）：
    smart_campus_audit.md              人读审计报告
    smart_campus_images_manifest.csv   逐图：filename,width,height,sha256,label_sha256,bbox_count,classes_present 等
    smart_campus_labels_manifest.csv   逐 label 文件：行数、空行、非法行、类分布、对应图是否存在
    smart_campus_class_stats.csv       逐类：图片数 / bbox 数（分 split）
    smart_campus_audit.json            结构化总记录（含 duplicate / near-duplicate 明细）

设计原则：
  * 不做任何猜测式修复：缺 label 就是 MISSING，非法行照原样记下并计非法，绝不改写数据集。
  * 不依赖第三方哈希包（imagehash 未安装）：dHash/pHash 自行实现。
  * 只读数据集目录；产物写到 --out（默认与数据集目录分开，避免污染原始数据）。

用法：
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe \
    docs/evidence/v3_domain_adaptation/_scripts/audit_smart_campus.py \
    --dataset-root data/v3_external/smart_campus/raw/<导出目录> \
    --out data/v3_external/smart_campus/audit
"""
import argparse
import collections
import csv
import hashlib
import json
import pathlib
import re
import sys
import time

import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "valid", "test")
RF_SUFFIX = re.compile(r"\.rf\.[0-9a-f]{8,}\.", re.I)          # Roboflow 导出会插 <name>.rf.<hash>
RF_SPLITISH = re.compile(r"\.(jpg|jpeg|png)\.rf\.", re.I)


# ---------------------------------------------------------------- 基础工具
def sha256_file(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def md5_file(p, chunk=1 << 20):
    h = hashlib.md5()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _bits_to_hex(bits):
    """把 0/1 的 numpy 数组按 64 位打包成 16 位十六进制（用于落盘与人工比对）。"""
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return "%016x" % v


def dhash_from_gray(gray8x9):
    """dHash：9x8 灰度，逐行比较相邻像素，得 64 位。"""
    bits = (gray8x9[:, 1:] > gray8x9[:, :-1]).flatten()
    return bits, _bits_to_hex(bits)


def phash_from_gray(gray32):
    """pHash：32x32 灰度 → DCT → 左上 8x8（去 DC）→ 中位数阈值，得 64 位。"""
    import cv2
    d = cv2.dct(gray32.astype(np.float32))
    low = d[:8, :8].flatten()
    med = np.median(low[1:])                       # 去 DC 后取中位数
    bits = (low > med).astype(np.uint8)
    return bits, _bits_to_hex(bits)


def perceptual_hashes(path):
    """返回 (dhash_hex, phash_hex, dhash_bits, phash_bits)；读不动返回 (None,None,None,None)。"""
    from PIL import Image
    try:
        with Image.open(path) as im:
            # JPEG 用 draft 加速（只解到需要的尺寸），失败再退回全解
            try:
                im.draft("L", (64, 64))
            except Exception:
                pass
            g = im.convert("L")
            g9 = np.asarray(g.resize((9, 8), Image.BILINEAR), dtype=np.int32)
            g32 = np.asarray(g.resize((32, 32), Image.BILINEAR), dtype=np.float32)
    except Exception:
        return None, None, None, None
    dh_bits, dh_hex = dhash_from_gray(g9)
    ph_bits, ph_hex = phash_from_gray(g32)
    return dh_hex, ph_hex, dh_bits, ph_bits


def read_data_yaml(root):
    """尽量不依赖 pyyaml：能找到就用 yaml，找不到就用正则抠 names。原始文本一并记录。"""
    info = {"path": None, "raw": None, "sha256": None, "names": None, "nc": None,
            "names_source": None}
    for cand in (root / "data.yaml", root / "data.yml", root / "dataset.yaml"):
        if cand.exists():
            raw = cand.read_text(encoding="utf-8", errors="replace")
            info.update({"path": str(cand), "raw": raw, "sha256": sha256_file(cand)})
            try:
                import yaml
                y = yaml.safe_load(raw)
                info["names"] = list(y.get("names")) if y.get("names") is not None else None
                info["nc"] = y.get("nc")
                info["names_source"] = "yaml"
            except Exception:
                m = re.search(r"names\s*:\s*\[(.*?)\]", raw, re.S)
                if m:
                    info["names"] = [x.strip().strip("'\"") for x in m.group(1).split(",") if x.strip()]
                    info["names_source"] = "regex"
                else:
                    m = re.search(r"names\s*:\s*\n((?:\s*-\s*.+\n?)+)", raw)
                    if m:
                        info["names"] = [x.split("-", 1)[1].strip().strip("'\"")
                                         for x in m.group(1).splitlines() if "-" in x]
                        info["names_source"] = "regex-block"
            break
    return info


def parse_label(path, nc):
    """解析一个 YOLO label 文件。返回 (records, stats)；绝不修改原文件。"""
    st = {"lines": 0, "empty_lines": 0, "bad_field_count": 0, "bad_class": 0,
          "bad_range": 0, "nonpositive_wh": 0, "coord_out_of_range": 0,
          "class_ids": collections.Counter(), "boxes": []}
    if not path.exists():
        return st["boxes"], st
    txt = path.read_text(encoding="utf-8", errors="replace")
    for i, line in enumerate(txt.splitlines(), 1):
        s = line.strip()
        if not s:
            st["empty_lines"] += 1
            continue
        st["lines"] += 1
        parts = s.split()
        if len(parts) < 5:
            st["bad_field_count"] += 1
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, w, h = (float(x) for x in parts[1:5])
        except ValueError:
            st["bad_field_count"] += 1
            continue
        if nc is not None and not (0 <= cls < nc):
            st["bad_class"] += 1
        if w <= 0 or h <= 0:
            st["nonpositive_wh"] += 1
        # 归一化坐标是否越界：允许 1e-6 的浮点误差
        x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        if min(x1, y1) < -1e-6 or max(x2, y2) > 1 + 1e-6:
            st["coord_out_of_range"] += 1
        elif not (0 <= cx <= 1 and 0 <= cy <= 1):
            st["bad_range"] += 1
        st["class_ids"][cls] += 1
        st["boxes"].append((cls, cx, cy, w, h, i))
    return st["boxes"], st


def near_dup_pairs(bits, max_dist, chunk=512):
    """两两比较 64 位哈希的汉明距离。bits: (n,) uint64。返回 [(i,j,dist), ...] i<j。"""
    n = len(bits)
    pairs = []
    bc = getattr(np, "bitwise_count", None)
    for i0 in range(0, n, chunk):
        a = bits[i0:i0 + chunk][:, None]
        for j0 in range(i0, n, chunk):
            b = bits[j0:j0 + chunk][None, :]
            x = np.bitwise_xor(a, b)
            d = bc(x).astype(np.int16) if bc is not None else np.array(
                [[bin(int(v)).count("1") for v in row] for row in x], dtype=np.int16)
            for p, q in np.argwhere(d <= max_dist):
                gi, gj = i0 + int(p), j0 + int(q)
                if gj <= gi:
                    continue
                pairs.append((gi, gj, int(d[p, q])))
    return pairs


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--out", default=None, help="默认 <dataset-root>/../audit")
    ap.add_argument("--near-dup-max-dist", type=int, default=2)
    ap.add_argument("--no-perceptual", action="store_true",
                    help="跳过 dHash/pHash（图很大时可省时间，但会缺近似重复分析）")
    a = ap.parse_args()

    root = pathlib.Path(a.dataset_root).resolve()
    if not root.is_dir():
        print("[FATAL] 数据集目录不存在：%s" % root)
        return 2
    out = pathlib.Path(a.out).resolve() if a.out else root.parent / "audit"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    dy = read_data_yaml(root)
    names = dy["names"]
    nc = dy["nc"] if dy["nc"] is not None else (len(names) if names else None)
    print("[info] data.yaml: %s | names_source=%s | nc=%s" % (dy["path"], dy["names_source"], nc))
    print("[info] names = %s" % (names,))

    # ---- 1. 枚举图片与 label
    rows = []
    label_paths = {}
    per_split = collections.Counter()
    for sp in SPLITS:
        idir, ldir = root / sp / "images", root / sp / "labels"
        if not idir.is_dir():
            continue
        for lp in sorted(ldir.glob("*.txt")) if ldir.is_dir() else []:
            label_paths.setdefault(lp.stem, lp)
        for p in sorted(idir.iterdir()):
            if p.suffix.lower() not in IMG_EXT:
                continue
            per_split[sp] += 1
            rows.append({"split": sp, "path": p, "stem": p.stem})
    # 顶层直铺（有些导出没有 split 子目录）也要兜住
    if not rows:
        for idir in sorted(root.glob("images")):
            for p in sorted(idir.iterdir()):
                if p.suffix.lower() in IMG_EXT:
                    per_split["(root)"] += 1
                    rows.append({"split": "(root)", "path": p, "stem": p.stem})
        for lp in sorted((root / "labels").glob("*.txt")) if (root / "labels").is_dir() else []:
            label_paths.setdefault(lp.stem, lp)

    print("[info] 枚举到图片 %d 张，label 文件 %d 个，分 split: %s"
          % (len(rows), len(label_paths), dict(per_split)))
    if not rows:
        print("[FATAL] 在 %s 下没找到任何图片——先确认导出布局" % root)
        return 2

    # ---- 2. 逐图处理
    img_rows = []
    class_img = collections.Counter()          # 类 -> 出现该类的图片数（全量）
    class_box = collections.Counter()
    class_box_split = collections.defaultdict(collections.Counter)
    class_img_split = collections.defaultdict(collections.Counter)
    bad = collections.Counter()
    dh_list, ph_list, dh_hex, ph_hex = [], [], [], []
    for k, r in enumerate(rows, 1):
        if k % 500 == 0:
            print("    ... %d/%d  (%.0fs)" % (k, len(rows), time.time() - t0))
        p = r["path"]
        lp = label_paths.get(r["stem"])
        boxes, st = parse_label(lp, nc) if lp else (None, None)
        # 尺寸
        w = h = None
        try:
            from PIL import Image
            with Image.open(p) as im:
                w, h = im.size
        except Exception as e:
            bad["image_unreadable"] += 1
        if a.no_perceptual:
            d_h = p_h = None
        else:
            d_h, p_h, db, pb = perceptual_hashes(p)
            if d_h is None:
                bad["hash_failed"] += 1
            else:
                dh_hex.append(d_h)
                ph_hex.append(p_h)
                dh_list.append(db)
                ph_list.append(pb)
        classes = sorted(set(st["class_ids"])) if st else []
        if lp is None:
            bad["missing_label"] += 1
        elif st["lines"] == 0 and st["empty_lines"] >= 0:
            bad["empty_label"] += 1
        if st is not None:
            for c, n in st["class_ids"].items():
                class_box[c] += n
                class_box_split[r["split"]][c] += n
            for c in classes:
                class_img[c] += 1
                class_img_split[r["split"]][c] += 1
        img_rows.append({
            "split": r["split"], "filename": p.name, "stem": r["stem"],
            "rel_path": str(p.relative_to(root)).replace("\\", "/"),
            "width": w, "height": h,
            "sha256": sha256_file(p), "md5": md5_file(p),
            "label_file": lp.name if lp else "",
            "label_sha256": sha256_file(lp) if lp else "MISSING",
            "bbox_count": st["lines"] if st else "",
            "classes_present": "|".join(str(c) for c in classes),
            "classes_present_names": "|".join(names[c] if names and c < len(names) else str(c)
                                             for c in classes),
            "empty_label": int(st is not None and st["lines"] == 0),
            "label_bad_lines": (st["bad_field_count"] + st["bad_class"] + st["bad_range"]
                                + st["nonpositive_wh"]) if st else "",
            "bbox_out_of_range": st["coord_out_of_range"] if st else "",
            "rf_export_name": int(bool(RF_SUFFIX.search(p.name))),
            "dhash": d_h, "phash": p_h,
        })

    # ---- 3. label 侧统计
    lab_rows = []
    for stem, lp in sorted(label_paths.items()):
        boxes, st = parse_label(lp, nc)
        has_img = any(r["stem"] == stem for r in rows)
        lab_rows.append({
            "label_file": lp.name, "stem": stem,
            "rel_path": str(lp.relative_to(root)).replace("\\", "/"),
            "sha256": sha256_file(lp),
            "n_boxes": st["lines"], "empty_lines": st["empty_lines"],
            "bad_field_count": st["bad_field_count"], "bad_class": st["bad_class"],
            "bad_range": st["bad_range"], "nonpositive_wh": st["nonpositive_wh"],
            "coord_out_of_range": st["coord_out_of_range"],
            "is_empty_label": int(st["lines"] == 0),
            "image_exists": int(has_img),
            "orphan_label": int(not has_img),
            "classes_present": "|".join(str(c) for c in sorted(st["class_ids"])),
        })

    # ---- 4. 精确重复（sha256 / md5）
    by_sha = collections.defaultdict(list)
    by_md5 = collections.defaultdict(list)
    for r in img_rows:
        by_sha[r["sha256"]].append(r["stem"])
        by_md5[r["md5"]].append(r["stem"])
    dup_sha = {k: v for k, v in by_sha.items() if len(v) > 1}
    dup_md5 = {k: v for k, v in by_md5.items() if len(v) > 1}
    # MD5 组与 SHA256 组在正常情况下应当一一对应；数量不一致本身就是异常信号，如实记下。
    md5_only = [k for k in dup_md5 if k not in
                {r["md5"] for r in img_rows if r["sha256"] in dup_sha}]

    # ---- 5. 近似重复（dHash / pHash）
    near = {"computed": not a.no_perceptual, "max_dist": a.near_dup_max_dist,
            "n_hashed": len(dh_list), "n_pairs_dhash": 0, "n_pairs_phash": 0,
            "n_pairs_min": 0, "n_pairs_min_cross_split": 0,
            "n_groups_min": 0, "largest_group": 0, "examples": []}
    if dh_list and not a.no_perceptual:
        # 直接用已经落盘的 16 位十六进制串还原 64 位整数（比再拼二进制串快且不会出错）
        arr_d = np.array([int(h, 16) for h in dh_hex], dtype=np.uint64)
        arr_p = np.array([int(h, 16) for h in ph_hex], dtype=np.uint64)
        pd = near_dup_pairs(arr_d, a.near_dup_max_dist)
        pp = near_dup_pairs(arr_p, a.near_dup_max_dist)
        near["n_pairs_dhash"] = len(pd)
        near["n_pairs_phash"] = len(pp)
        # min(dHash, pHash) <= 2 的合并判定
        best = {}
        for (i, j, d) in pd:
            best[(i, j)] = min(best.get((i, j), 99), d)
        for (i, j, d) in pp:
            best[(i, j)] = min(best.get((i, j), 99), d)
        keep = {k: v for k, v in best.items() if v <= a.near_dup_max_dist}
        near["n_pairs_min"] = len(keep)
        uf = UF(len(img_rows))
        cross = 0
        for (i, j), d in sorted(keep.items(), key=lambda kv: kv[1]):
            uf.union(i, j)
            if img_rows[i]["split"] != img_rows[j]["split"]:
                cross += 1
            if len(near["examples"]) < 40:
                near["examples"].append({"a": img_rows[i]["rel_path"], "b": img_rows[j]["rel_path"],
                                         "min_dist": d, "same_split": img_rows[i]["split"] == img_rows[j]["split"]})
        near["n_pairs_min_cross_split"] = cross
        comp = collections.defaultdict(list)
        for i in range(len(img_rows)):
            comp[uf.find(i)].append(i)
        sizes = sorted((len(v) for v in comp.values() if len(v) > 1), reverse=True)
        near["n_groups_min"] = len(sizes)
        near["largest_group"] = sizes[0] if sizes else 0
        near["group_size_hist"] = dict(collections.Counter(sizes))

    # ---- 6. 类别统计
    cls_rows = []
    all_cls = sorted(set(list(class_box) + list(class_img)))
    for c in all_cls:
        nm = names[c] if names and 0 <= c < len(names) else "(unnamed:%d)" % c
        row = {"class_id": c, "class_name": nm,
               "images_total": class_img.get(c, 0), "boxes_total": class_box.get(c, 0)}
        for sp in list(SPLITS) + (["(root)"] if per_split.get("(root)") else []):
            row["images_%s" % sp] = class_img_split[sp].get(c, 0)
            row["boxes_%s" % sp] = class_box_split[sp].get(c, 0)
        cls_rows.append(row)

    # ---- 7. 落盘
    def wcsv(name, dicts, fields=None):
        path = out / name
        if not dicts:
            path.write_text("", encoding="utf-8")
            return path
        fields = fields or list(dicts[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(dicts)
        return path

    wcsv("smart_campus_images_manifest.csv", img_rows)
    wcsv("smart_campus_labels_manifest.csv", lab_rows)
    wcsv("smart_campus_class_stats.csv", cls_rows)

    # ---- 8. 报告
    n_img = len(img_rows)
    n_lab = len(lab_rows)
    n_box = sum(r["bbox_count"] for r in img_rows if isinstance(r["bbox_count"], int))
    oob_imgs = sum(1 for r in img_rows if isinstance(r["bbox_out_of_range"], int) and r["bbox_out_of_range"] > 0)
    rf_named = sum(r["rf_export_name"] for r in img_rows)
    empty_lab = sum(1 for r in lab_rows if r["is_empty_label"])
    orphan = sum(1 for r in lab_rows if r["orphan_label"])
    missing_lab = sum(1 for r in img_rows if r["label_sha256"] == "MISSING")

    summary = {
        "dataset_root": str(root),
        "audit_time_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_s": round(time.time() - t0, 1),
        "data_yaml": {k: dy[k] for k in ("path", "sha256", "names", "nc", "names_source")},
        "counts": {"images": n_img, "label_files": n_lab, "bboxes": n_box,
                   "per_split_images": dict(per_split)},
        "problems": {"images_missing_label": missing_lab, "empty_labels": empty_lab,
                     "orphan_labels": orphan, "images_with_oob_bbox": oob_imgs,
                     "roboflow_style_filenames": rf_named,
                     "unreadable_images": bad["image_unreadable"],
                     "hash_failed": bad["hash_failed"]},
        "label_parse": {
            "bad_field_count": sum(r["bad_field_count"] for r in lab_rows),
            "bad_class": sum(r["bad_class"] for r in lab_rows),
            "bad_range": sum(r["bad_range"] for r in lab_rows),
            "nonpositive_wh": sum(r["nonpositive_wh"] for r in lab_rows),
            "coord_out_of_range_boxes": sum(r["coord_out_of_range"] for r in lab_rows),
        },
        "duplicates": {
            "exact_sha256_groups": len(dup_sha),
            "exact_sha256_images_in_groups": sum(len(v) for v in dup_sha.values()),
            "exact_md5_groups": len(dup_md5),
            "md5_groups_without_matching_sha256_group": len(md5_only),
            "sha256_groups_detail": [{"sha256": k, "stems": v} for k, v in
                                     sorted(dup_sha.items(), key=lambda kv: -len(kv[1]))[:50]],
        },
        "near_duplicates": near,
        "class_stats": cls_rows,
    }
    (out / "smart_campus_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    L = []
    L.append("# Smart Campus Dataset 完整性审计")
    L.append("")
    L.append("- 数据集目录：`%s`" % root)
    L.append("- 审计时间：`%s`（耗时 %.0f s）" % (summary["audit_time_local"], summary["elapsed_s"]))
    L.append("- data.yaml：`%s`（SHA256 `%s`，names 来源 `%s`）"
             % (dy["path"], (dy["sha256"] or "")[:16], dy["names_source"]))
    L.append("- 类别：%s" % (names,))
    L.append("")
    L.append("## 1. 规模")
    L.append("")
    L.append("| 项 | 值 |")
    L.append("|---|---|")
    L.append("| 图片数 | %d |" % n_img)
    L.append("| label 文件数 | %d |" % n_lab)
    L.append("| bbox 总数 | %d |" % n_box)
    L.append("| 各 split 图片数 | %s |" % dict(per_split))
    L.append("")
    L.append("## 2. 完整性问题")
    L.append("")
    L.append("| 检查 | 数量 |")
    L.append("|---|---|")
    L.append("| 有图无 label | **%d** |" % missing_lab)
    L.append("| 空 label 文件（0 框） | %d |" % empty_lab)
    L.append("| orphan label（有 label 无图） | **%d** |" % orphan)
    L.append("| 含越界 bbox 的图片 | **%d** |" % oob_imgs)
    L.append("| Roboflow 风格文件名（`.rf.<hash>`） | %d |" % rf_named)
    L.append("| 读不出的图片 | %d |" % bad["image_unreadable"])
    L.append("| label 字段数非法行 | %d |" % summary["label_parse"]["bad_field_count"])
    L.append("| label 类别越界行 | %d |" % summary["label_parse"]["bad_class"])
    L.append("| label 坐标越界框 | %d |" % summary["label_parse"]["coord_out_of_range_boxes"])
    L.append("| label w/h 非正 | %d |" % summary["label_parse"]["nonpositive_wh"])
    L.append("")
    L.append("## 3. 重复情况")
    L.append("")
    L.append("- **exact duplicate**：SHA256 完全相同的组 **%d** 组、涉及 **%d** 张图"
             % (len(dup_sha), sum(len(v) for v in dup_sha.values())))
    if dup_sha:
        L.append("")
        for k, v in sorted(dup_sha.items(), key=lambda kv: -len(kv[1]))[:10]:
            L.append("  - `%s…` × %d：%s" % (k[:12], len(v), ", ".join(v[:8])))
    if near["computed"]:
        L.append("- **near duplicate**（阈值 min(dHash,pHash) ≤ %d）："
                 % near["max_dist"])
        L.append("  - dHash 命中 %d 对；pHash 命中 %d 对；取 min 后命中 **%d 对**"
                 % (near["n_pairs_dhash"], near["n_pairs_phash"], near["n_pairs_min"]))
        L.append("  - 其中**跨 split** 的有 **%d 对**（这正是 Roboflow 自带划分可能泄漏的证据）"
                 % near["n_pairs_min_cross_split"])
        L.append("  - 合并成 **%d** 个组，最大组 %d 张；组大小分布 %s"
                 % (near["n_groups_min"], near["largest_group"], near.get("group_size_hist")))
    L.append("")
    L.append("## 4. 逐类统计")
    L.append("")
    L.append("| id | 类名 | 图片数(总) | bbox 数(总) | 图片(train/valid/test) | bbox(train/valid/test) |")
    L.append("|---|---|---|---|---|---|")
    for r in cls_rows:
        L.append("| %d | %s | %d | %d | %d/%d/%d | %d/%d/%d |"
                 % (r["class_id"], r["class_name"], r["images_total"], r["boxes_total"],
                    r.get("images_train", 0), r.get("images_valid", 0), r.get("images_test", 0),
                    r.get("boxes_train", 0), r.get("boxes_valid", 0), r.get("boxes_test", 0)))
    L.append("")
    L.append("## 5. 产物")
    L.append("")
    L.append("- `smart_campus_images_manifest.csv`（%d 行，含 sha256/md5/dhash/phash）" % n_img)
    L.append("- `smart_campus_labels_manifest.csv`（%d 行）" % n_lab)
    L.append("- `smart_campus_class_stats.csv`（%d 行）" % len(cls_rows))
    L.append("- `smart_campus_audit.json`（结构化明细）")
    L.append("")
    L.append("> 本审计**只读**数据集，未修改任何原始文件。")
    (out / "smart_campus_audit.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("[OK] 产物写到 %s" % out)
    print("     图片 %d / label %d / bbox %d / exact-dup 组 %d / near-dup 对 %d"
          % (n_img, n_lab, n_box, len(dup_sha), near.get("n_pairs_min", 0)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
