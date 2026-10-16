"""构建 Smart Campus V3 的 train/val/test 划分（**组级、确定性、可冻结**）。

数据流：
  1. 池 = 含任一映射类（cracks/corrosion/paint_degradation）的图，且不在筛除名单里；
  2. 约束组 = union-find 合并四类关系：
       exact      —— sha256 相同（字节级重复）
       hash_near  —— min(dHash,pHash) <= --max-dist（用户指定的硬约束）
       mirror     —— 水平翻转后 min(dHash,pHash) <= --mirror-max-dist（镜像同源）
       same_view  —— min(dHash,pHash) <= --conservative-dist **且宽高完全相同**（保守同视角）
  3. 组级分配 70/15/15：组是不可分割单元，按组大小降序 + 固定 seed 决定顺序，
     每次把组放进「当前完成度最低」的 split，最后做类别覆盖修补；
  4. 产物：split 清单、重映射后的 label（class id 按 MBDD 顺序 0=peeling/1=crack/2=rust，
     未映射类别的框被丢弃）、逐图 manifest、constraint group 台账、split freeze JSON + SHA256。

**本脚本不改动原始数据集**：原始图与 label 只读，重映射 label 写到新的 V3 目录。

用法（先看不写）：
  ... build_v3_split.py --dataset-root ... --images-manifest ... --exclusions-dir ... --out ...
加 --write 才落盘并冻结。
"""
import argparse
import collections
import csv
import hashlib
import json
import pathlib
import sys
import time

import cv2
import numpy as np

# 映射：Smart Campus 类名 -> MBDD 的 class id（顺序取自 data/splits_v2/data_orig.yaml：0=peeling,1=crack,2=rust）
MAP = {"paint_degradation": 0, "cracks": 1, "corrosion": 2}
SRC_NAMES = ["corrosion", "cracks", "paint_degradation", "potholes", "spalling"]
SPLITS = ("train", "val", "test")
RATIO = {"train": 0.70, "val": 0.15, "test": 0.15}
NL = chr(10)   # 换行（显式写出，避免转义踩坑）
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


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
            return True
        return False


def pairs_at(bits, max_dist, chunk=512):
    n = len(bits)
    out = []
    for i0 in range(0, n, chunk):
        a = bits[i0:i0 + chunk][:, None]
        for j0 in range(i0, n, chunk):
            b = bits[j0:j0 + chunk][None, :]
            d = np.bitwise_count(np.bitwise_xor(a, b)).astype(np.int16)
            for p, q in np.argwhere(d <= max_dist):
                gi, gj = i0 + int(p), j0 + int(q)
                if gj > gi:
                    out.append((gi, gj, int(d[p, q])))
    return out


def load_gray32(root, rel):
    im = cv2.imread(str(root / rel), cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None
    return cv2.resize(im, (32, 32), interpolation=cv2.INTER_AREA)


def hashes_from_g32(g):
    g9 = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int32)
    bits = (g9[:, 1:] > g9[:, :-1]).flatten()
    d = 0
    for b in bits:
        d = (d << 1) | int(b)
    low = cv2.dct(g.astype(np.float32))[:8, :8].flatten()
    med = np.median(low[1:])
    p = 0
    for b in (low > med).astype(np.uint8):
        p = (p << 1) | int(b)
    return np.uint64(d), np.uint64(p)


def sha256_file(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def parse_boxes(lp, nc):
    """返回 [(class_id, cx, cy, w, h)]；坏行跳过（调用方已由审计确认 0 坏行）。"""
    out = []
    if not lp.exists():
        return out
    for line in lp.read_text(encoding="utf-8", errors="replace").splitlines():
        q = line.split()
        if len(q) < 5:
            continue
        try:
            c = int(float(q[0]))
            x = [float(v) for v in q[1:5]]
        except ValueError:
            continue
        if 0 <= c < nc:
            out.append((c, *x))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--images-manifest", required=True)
    ap.add_argument("--exclusions-dir", required=True,
                    help="含 excluded_<class>.csv 与 manual_visual_flags__<class>.csv 的目录")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-dist", type=int, default=2)
    ap.add_argument("--mirror-max-dist", type=int, default=2)
    ap.add_argument("--conservative-dist", type=int, default=6)
    ap.add_argument("--min-boxes-val-test", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--nc", type=int, default=5)
    a = ap.parse_args()

    root = pathlib.Path(a.dataset_root).resolve()
    out = pathlib.Path(a.out).resolve()
    edir = pathlib.Path(a.exclusions_dir).resolve()
    rows = list(csv.DictReader(open(a.images_manifest, encoding="utf-8")))
    by_stem = {r["stem"]: r for r in rows}

    # ---------- 1. 池 ----------
    ex = {}
    for cls in MAP:
        s = set()
        f1 = edir / ("excluded_%s.csv" % cls)
        if f1.exists():
            s |= {r["stem"] for r in csv.DictReader(open(f1, encoding="utf-8"))}
        f2 = edir / ("manual_visual_flags__%s.csv" % cls)
        if f2.exists():
            s |= {r["stem"] for r in csv.DictReader(open(f2, encoding="utf-8"))}
        ex[cls] = s
    excluded_any = ex["corrosion"] | ex["cracks"] | ex["paint_degradation"]

    pool, drop_unmapped, drop_excluded = [], 0, 0
    for r in rows:
        present = [p for p in r["classes_present_names"].split("|") if p]
        mapped = [p for p in present if p in MAP]
        if r["stem"] in excluded_any:
            drop_excluded += 1
            continue
        if not mapped:
            drop_unmapped += 1
            continue
        pool.append(r)
    print("[池] 总 %d 张 → 剔除名单 %d、只含未映射类 %d → **池 %d 张**"
          % (len(rows), drop_excluded, drop_unmapped, len(pool)))

    n = len(pool)
    idx = {r["stem"]: i for i, r in enumerate(pool)}
    sha = np.array([int(r["sha256"], 16) for r in pool], dtype=object)
    dh = np.array([int(r["dhash"], 16) for r in pool], dtype=np.uint64)
    ph = np.array([int(r["phash"], 16) for r in pool], dtype=np.uint64)
    wh = [(r["width"], r["height"]) for r in pool]

    uf = UF(n)
    reason = collections.defaultdict(set)     # (i,j) -> {relations}
    stats = collections.Counter()

    # exact
    g = collections.defaultdict(list)
    for i, r in enumerate(pool):
        g[r["sha256"]].append(i)
    for k, v in g.items():
        for j in v[1:]:
            uf.union(v[0], j)
            stats["exact_pairs"] += 1
    print("[约束] exact(sha256) 组 %d，涉及 %d 张"
          % (len([1 for v in g.values() if len(v) > 1]),
             sum(len(v) for v in g.values() if len(v) > 1)))

    # hash near
    pn = pairs_at(dh, a.max_dist)
    pnp = pairs_at(ph, a.max_dist)
    best = {}
    for (i, j, d) in pn:
        best[(i, j)] = [d, 99]
    for (i, j, d) in pnp:
        if (i, j) in best:
            best[(i, j)][1] = d
        else:
            best[(i, j)] = [99, d]
    near = {k: v for k, v in best.items() if min(v) <= a.max_dist}
    for k in near:
        uf.union(*k)
    stats["hash_near_pairs"] = len(near)
    print("[约束] hash_near(min<=%d) 候选对 %d" % (a.max_dist, len(near)))

    # mirror
    fl_d, fl_p = [], []
    for r in pool:
        g32 = load_gray32(root, r["rel_path"])
        if g32 is None:
            fl_d.append(np.uint64(0)); fl_p.append(np.uint64(0)); continue
        fd, fp = hashes_from_g32(cv2.flip(g32, 1))
        fl_d.append(fd); fl_p.append(fp)
    all_d = np.concatenate([dh, np.array(fl_d, dtype=np.uint64)])
    all_p = np.concatenate([ph, np.array(fl_p, dtype=np.uint64)])
    md = pairs_at(all_d, a.mirror_max_dist)
    mp = pairs_at(all_p, a.mirror_max_dist)
    bm = {}
    for (i, j, d) in md:
        bm[(i, j)] = [d, 99]
    for (i, j, d) in mp:
        if (i, j) in bm:
            bm[(i, j)][1] = d
        else:
            bm[(i, j)] = [99, d]
    mirror = [k for k, v in bm.items()
              if min(v) <= a.mirror_max_dist and (k[0] < n) != (k[1] < n)]
    for i, j in mirror:
        ni, nj = (i, j - n) if i < n else (j, i - n)
        uf.union(ni, nj)
    stats["mirror_pairs"] = len(mirror)
    print("[约束] mirror(翻转后 min<=%d) 候选对 %d" % (a.mirror_max_dist, len(mirror)))

    # conservative same-view：更松的阈值 + 尺寸完全相同
    sameview = []
    if a.conservative_dist > a.max_dist:
        # 必须**单独**在更松的阈值上重新求对：严格阈值那份 best 里根本不含 3..conservative 的对
        pdc = pairs_at(dh, a.conservative_dist)
        ppc = pairs_at(ph, a.conservative_dist)
        bc = {}
        for (i, j, d) in pdc:
            bc[(i, j)] = [d, 99]
        for (i, j, d) in ppc:
            if (i, j) in bc:
                bc[(i, j)][1] = d
            else:
                bc[(i, j)] = [99, d]
        for (i, j), v in bc.items():
            if a.max_dist < min(v) <= a.conservative_dist and wh[i] == wh[j]:
                sameview.append((i, j))
        for i, j in sameview:
            uf.union(i, j)
        stats["same_view_pairs"] = len(sameview)
        print("[约束] same_view(min<=%d 且同尺寸) 追加 %d 对"
              % (a.conservative_dist, len(sameview)))

    # ---------- 2. 组 ----------
    grp = collections.defaultdict(list)
    for i in range(n):
        grp[uf.find(i)].append(i)
    groups = sorted(grp.values(), key=lambda m: (-len(m), pool[m[0]]["stem"]))
    sizes = [len(m) for m in groups]
    print("[组] %d 组；最大 %d 张（占池 %.2f%%）；单张组 %d 个"
          % (len(groups), sizes[0], 100.0 * sizes[0] / n, sum(1 for s in sizes if s == 1)))

    # 每组的类别框数（用重映射后的框）
    def group_boxes(mem):
        c = collections.Counter()
        for i in mem:
            r = pool[i]
            lp = root / r["rel_path"].replace("/images/", "/labels/").rsplit(".", 1)[0]
            lp = pathlib.Path(str(lp) + ".txt")
            for (cid, *_rest) in parse_boxes(lp, a.nc):
                nm = SRC_NAMES[cid] if cid < len(SRC_NAMES) else None
                if nm in MAP:
                    c[MAP[nm]] += 1
        return c

    gboxes = [group_boxes(m) for m in groups]

    # ---------- 3. 分配 ----------
    rng = np.random.default_rng(a.seed)
    order = list(range(len(groups)))
    # 同尺寸组之间用 seed 打乱顺序（组大小降序为主键，保持确定性）
    by_size = collections.defaultdict(list)
    for gi in order:
        by_size[len(groups[gi])].append(gi)
    ordered = []
    for sz in sorted(by_size, reverse=True):
        lst = by_size[sz]
        perm = rng.permutation(len(lst))
        ordered += [lst[p] for p in perm]

    target = {s: RATIO[s] * n for s in SPLITS}
    cnt = {s: 0 for s in SPLITS}
    assign = {}
    for gi in ordered:
        # 放进「当前完成度 cnt/target 最低」的 split
        s = min(SPLITS, key=lambda k: (cnt[k] / target[k], SPLITS.index(k)))
        assign[gi] = s
        cnt[s] += len(groups[gi])

    # 类别覆盖修补：val/test 每类框数不足则从 train 搬最小的含该类组
    repairs = []
    for s in ("val", "test"):
        for cls_id in sorted(MAP.values()):
            have = sum(gboxes[gi][cls_id] for gi in range(len(groups)) if assign[gi] == s)
            if have >= a.min_boxes_val_test:
                continue
            cands = [(len(groups[gi]), gi) for gi in range(len(groups))
                     if assign[gi] == "train" and gboxes[gi][cls_id] > 0]
            cands.sort()
            moved = 0
            for _sz, gi in cands:
                if have >= a.min_boxes_val_test:
                    break
                assign[gi] = s
                cnt["train"] -= len(groups[gi])
                cnt[s] += len(groups[gi])
                have += gboxes[gi][cls_id]
                moved += 1
            repairs.append({"split": s, "class_id": cls_id, "moved_groups": moved,
                            "boxes_after": have})
    if repairs:
        print("[修补] 类别覆盖修补：%s" % repairs)

    # ---------- 4. 诊断 ----------
    def split_boxes(s):
        c = collections.Counter()
        for gi in range(len(groups)):
            if assign[gi] == s:
                c.update(gboxes[gi])
        return c

    diag = {}
    for s in SPLITS:
        c = split_boxes(s)
        diag[s] = {"images": cnt[s], "groups": sum(1 for gi in range(len(groups)) if assign[gi] == s),
                   "boxes": dict(c)}
    print("[分配] train %d / val %d / test %d 张（目标 %.1f/%.1f/%.1f）"
          % (cnt["train"], cnt["val"], cnt["test"], target["train"], target["val"], target["test"]))
    for s in SPLITS:
        print("        %-5s boxes by class: %s" % (s, diag[s]["boxes"]))

    # 跨 split 检查：约束组内不可能跨（组是不可分割单元），但显式验证一遍
    cross_near = cross_mirror = cross_exact = 0
    # uf.find 返回的是「代表图的下标」，不是组号；先建立映射再查
    root2gi = {}
    for gi, m in enumerate(groups):
        root2gi[uf.find(m[0])] = gi
    split_of = {i: assign[root2gi[uf.find(i)]] for i in range(n)}
    for (i, j) in near:
        if split_of[i] != split_of[j]:
            cross_near += 1
    for i, j in mirror:
        ni, nj = (i, j - n) if i < n else (j, i - n)
        if split_of[ni] != split_of[nj]:
            cross_mirror += 1
    for k, v in g.items():
        if len(v) > 1 and len({split_of[i] for i in v}) > 1:
            cross_exact += 1
    print("[检查] 跨 split：exact %d、hash_near %d、mirror %d（应全为 0）"
          % (cross_exact, cross_near, cross_mirror))

    summary = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dataset_root": str(root),
        "params": {"max_dist": a.max_dist, "mirror_max_dist": a.mirror_max_dist,
                   "conservative_dist": a.conservative_dist, "seed": a.seed,
                   "ratio": RATIO, "min_boxes_val_test": a.min_boxes_val_test,
                   "class_map": {"paint_degradation": 0, "cracks": 1, "corrosion": 2},
                   "class_names_mbdd_order": ["peeling", "crack", "rust"]},
        "pool": {"n_images": n, "dropped_excluded": drop_excluded,
                 "dropped_unmapped_only": drop_unmapped},
        "constraints": {"n_groups": len(groups), "max_group": sizes[0],
                        "max_group_share_pct": round(100.0 * sizes[0] / n, 3),
                        "singleton_groups": sum(1 for s in sizes if s == 1),
                        "group_size_hist": dict(collections.Counter(sizes)),
                        "pairs": dict(stats)},
        "splits": diag,
        "repairs": repairs,
        "cross_split_violations": {"exact": cross_exact, "hash_near": cross_near,
                                   "mirror": cross_mirror},
    }

    if a.write:
        out.mkdir(parents=True, exist_ok=True)

        # ---- (1) 重映射 label：class id 换成 MBDD 顺序（0=peeling,1=crack,2=rust），
        #          未映射类别（potholes/spalling）的框丢弃。原始 label 不动。----
        lab_dir = out / "labels_remapped"
        lab_dir.mkdir(parents=True, exist_ok=True)
        drop_box_stats = collections.Counter()
        lab_sha = {}
        keep_box = collections.Counter()
        for r in pool:
            src = pathlib.Path(str(root / r["rel_path"].replace("/images/", "/labels/")).rsplit(".", 1)[0] + ".txt")
            kept_lines = []
            for (cid, cx, cy, bw, bh) in parse_boxes(src, a.nc):
                nm = SRC_NAMES[cid]
                if nm in MAP:
                    kept_lines.append("%d %.6f %.6f %.6f %.6f" % (MAP[nm], cx, cy, bw, bh))
                    keep_box[nm] += 1
                else:
                    drop_box_stats[nm] += 1
            dst = lab_dir / (r["stem"] + ".txt")
            dst.write_text(NL.join(kept_lines) + (NL if kept_lines else ""), encoding="utf-8")
            lab_sha[r["stem"]] = sha256_file(dst)
        summary["labels_remapped"] = {
            "n_files": len(lab_sha), "boxes_kept_by_source_class": dict(keep_box),
            "boxes_dropped_by_source_class": dict(drop_box_stats)}
        print("[写] 重映射 label %d 个；保留框 %s；丢弃未映射框 %s"
              % (len(lab_sha), dict(keep_box), dict(drop_box_stats)))

        # ---- (2) split 清单 ----
        for s in SPLITS:
            lst = sorted(pool[i]["rel_path"] for i in range(n) if split_of[i] == s)
            (out / ("smart_campus_v3_%s.txt" % s)).write_text(NL.join(lst) + NL, encoding="utf-8")

        # ---- (3) constraint group 台账 ----
        with open(out / "smart_campus_v3_duplicate_groups.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["group_id", "size", "final_split", "members_stems", "members_source_splits"])
            for gi, m in enumerate(groups):
                w.writerow([gi, len(m), assign[gi],
                            "|".join(pool[i]["stem"] for i in m),
                            "|".join(pool[i]["split"] for i in m)])

        # ---- (4) 逐图 manifest ----
        with open(out / "smart_campus_v3_split_manifest.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["stem", "final_split", "group_id", "group_size", "source_split",
                        "rel_path", "sha256", "dhash", "phash", "width", "height",
                        "mapped_classes", "unmapped_classes_dropped",
                        "label_remapped", "label_remapped_sha256"])
            for i, r in enumerate(pool):
                gi = root2gi[uf.find(i)]
                mapped = [p for p in r["classes_present_names"].split("|") if p in MAP]
                unmapped = [p for p in r["classes_present_names"].split("|") if p and p not in MAP]
                w.writerow([r["stem"], split_of[i], gi, len(groups[gi]), r["split"],
                            r["rel_path"], r["sha256"], r["dhash"], r["phash"],
                            r["width"], r["height"], "|".join(mapped), "|".join(unmapped),
                            "labels_remapped/%s.txt" % r["stem"], lab_sha[r["stem"]]])

        # ---- (5) test manifest（封存件）----
        test_rows = []
        for i, r in enumerate(pool):
            if split_of[i] != "test":
                continue
            test_rows.append({"stem": r["stem"], "rel_path": r["rel_path"], "sha256": r["sha256"],
                              "dhash": r["dhash"], "phash": r["phash"],
                              "label_remapped": "labels_remapped/%s.txt" % r["stem"],
                              "label_remapped_sha256": lab_sha[r["stem"]],
                              "group_id": root2gi[uf.find(i)]})
        test_rows.sort(key=lambda x: x["stem"])
        with open(out / "smart_campus_v3_test_manifest.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(test_rows[0].keys()))
            w.writeheader()
            w.writerows(test_rows)
        print("[写] test manifest：%d 行" % len(test_rows))

        json.dump(summary, open(out / "smart_campus_v3_split_build.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print("[写] 已写 label/split/组台账/manifest/test manifest/build JSON → %s" % out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
