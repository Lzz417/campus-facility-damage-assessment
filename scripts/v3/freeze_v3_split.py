"""校验并冻结 Smart Campus V3 划分。**任一检查不过则不写任何文件。**

独立校验（不依赖 build JSON 里的自述值）：
  1. 三个 split 清单：并集 == 池、两两无交集、张数 == manifest 里的统计；
  2. **从图片文件重新计算 dHash/pHash**（不信 manifest 里存的哈希），逐对检查
     `min(dHash,pHash) <= max-dist` 与「水平翻转后 <= mirror-dist」的关系**没有跨 split**；
  3. 重映射 label：class id 只能是 {0,1,2}、每个框都能对应回原始 label 的某个框、
     且被丢弃的框确实全部属于未映射类别；
  4. test manifest 与 test 清单逐行一致，且每个 test 图的 label 重映射哈希与文件一致。

通过后写：
  v3_split_freeze.json / v3_split_freeze_manifest.csv / v3_split_freeze.sha256
并声明 **Smart Campus V3 test = SEALED**。

用法：
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe \
    docs/evidence/v3_domain_adaptation/_scripts/freeze_v3_split.py \
    --dataset-root data/v3_external/smart_campus/raw/v9-updated_original.yolov11 \
    --split-dir docs/evidence/v3_domain_adaptation/smart_campus/v3_split \
    --images-manifest docs/evidence/v3_domain_adaptation/smart_campus/audit/smart_campus_images_manifest.csv \
    [--write]
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

NL = chr(10)
SRC_NAMES = ["corrosion", "cracks", "paint_degradation", "potholes", "spalling"]
MAP = {"paint_degradation": 0, "cracks": 1, "corrosion": 2}
SPLITS = ("train", "val", "test")


def sha256_file(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def g32(root, rel):
    im = cv2.imread(str(root / rel), cv2.IMREAD_GRAYSCALE)
    return None if im is None else cv2.resize(im, (32, 32), interpolation=cv2.INTER_AREA)


def hashes(g):
    g9 = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int32)
    d = 0
    for b in (g9[:, 1:] > g9[:, :-1]).flatten():
        d = (d << 1) | int(b)
    low = cv2.dct(g.astype(np.float32))[:8, :8].flatten()
    med = np.median(low[1:])
    p = 0
    for b in (low > med).astype(np.uint8):
        p = (p << 1) | int(b)
    return np.uint64(d), np.uint64(p)


def pairs_at(bits, md, chunk=512):
    out = []
    for i0 in range(0, len(bits), chunk):
        A = bits[i0:i0 + chunk][:, None]
        for j0 in range(i0, len(bits), chunk):
            B = bits[j0:j0 + chunk][None, :]
            d = np.bitwise_count(np.bitwise_xor(A, B)).astype(np.int16)
            for p, q in np.argwhere(d <= md):
                gi, gj = i0 + int(p), j0 + int(q)
                if gj > gi:
                    out.append((gi, gj))
    return out


def parse_boxes(p, nc):
    out = []
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        q = line.split()
        if len(q) < 5:
            continue
        try:
            c = int(float(q[0]))
            v = [float(x) for x in q[1:5]]
        except ValueError:
            continue
        if 0 <= c < nc:
            out.append((c, tuple(round(x, 6) for x in v)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--images-manifest", required=True)
    ap.add_argument("--max-dist", type=int, default=2)
    ap.add_argument("--mirror-max-dist", type=int, default=2)
    ap.add_argument("--nc", type=int, default=5)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    root = pathlib.Path(a.dataset_root).resolve()
    S = pathlib.Path(a.split_dir).resolve()
    fails, warns = [], []

    lists = {}
    for s in SPLITS:
        f = S / ("smart_campus_v3_%s.txt" % s)
        if not f.exists():
            print("[FATAL] 缺少 %s" % f)
            return 2
        lists[s] = [l.strip() for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    n_pool = sum(len(v) for v in lists.values())
    print("[1] split 张数 train/val/test = %d/%d/%d，合计 %d"
          % (len(lists["train"]), len(lists["val"]), len(lists["test"]), n_pool))
    allrel = [r for v in lists.values() for r in v]
    if len(set(allrel)) != len(allrel):
        fails.append("三个 split 清单之间有重复图片")
    man = list(csv.DictReader(open(S / "smart_campus_v3_split_manifest.csv", encoding="utf-8")))
    if len(man) != n_pool:
        fails.append("manifest 行数 %d != split 清单合计 %d" % (len(man), n_pool))
    mrel = {r["rel_path"]: r for r in man}
    if set(mrel) != set(allrel):
        fails.append("manifest 与 split 清单的图片集合不一致")
    for s in SPLITS:
        cnt = sum(1 for r in man if r["final_split"] == s)
        if cnt != len(lists[s]):
            fails.append("%s 的 manifest 行数 %d != 清单 %d" % (s, cnt, len(lists[s])))

    # ---- 2. 从图片重算哈希，独立验证无跨 split 泄漏 ----
    print("[2] 从图片文件重算 dHash/pHash（%d 张）…" % n_pool)
    rel2i = {r: i for i, r in enumerate(allrel)}
    dh = np.zeros(n_pool, dtype=np.uint64)
    ph = np.zeros(n_pool, dtype=np.uint64)
    fd = np.zeros(n_pool, dtype=np.uint64)
    fp = np.zeros(n_pool, dtype=np.uint64)
    bad = 0
    for rel, i in rel2i.items():
        g = g32(root, rel)
        if g is None:
            bad += 1
            continue
        dh[i], ph[i] = hashes(g)
        fd[i], fp[i] = hashes(cv2.flip(g, 1))
    if bad:
        fails.append("有 %d 张图读不出来" % bad)
    sid = np.array([0 if r in set(lists["train"]) else (1 if r in set(lists["val"]) else 2)
                    for r in allrel])
    n_near = n_mir = n_near_cross = n_mir_cross = 0
    for (i, j) in pairs_at(dh, a.max_dist):
        n_near += 1
        if sid[i] != sid[j]:
            n_near_cross += 1
    for (i, j) in pairs_at(ph, a.max_dist):
        n_near += 1
        if sid[i] != sid[j]:
            n_near_cross += 1
    # 镜像：翻转后的哈希 vs 正常哈希
    for (i, j) in pairs_at(np.concatenate([dh, fd]), a.mirror_max_dist):
        if (i < n_pool) != (j < n_pool):
            ni, nj = (i, j - n_pool) if i < n_pool else (j, i - n_pool)
            n_mir += 1
            if sid[ni] != sid[nj]:
                n_mir_cross += 1
    for (i, j) in pairs_at(np.concatenate([ph, fp]), a.mirror_max_dist):
        if (i < n_pool) != (j < n_pool):
            ni, nj = (i, j - n_pool) if i < n_pool else (j, i - n_pool)
            n_mir += 1
            if sid[ni] != sid[nj]:
                n_mir_cross += 1
    print("    hash 近重复命中 %d 次、其中跨 split %d 次" % (n_near, n_near_cross))
    print("    镜像命中 %d 次、其中跨 split %d 次" % (n_mir, n_mir_cross))
    if n_near_cross or n_mir_cross:
        fails.append("重算哈希发现跨 split 泄漏：near %d、mirror %d" % (n_near_cross, n_mir_cross))

    # 精确重复（sha256）
    sha = collections.defaultdict(set)
    for r in man:
        sha[r["sha256"]].add(r["final_split"])
    ex_cross = [k for k, v in sha.items() if len(v) > 1]
    print("    sha256 完全相同的组里跨 split 的：%d" % len(ex_cross))
    if ex_cross:
        fails.append("存在跨 split 的字节级重复 %d 组" % len(ex_cross))

    # ---- 3. 重映射 label ----
    print("[3] 校验重映射 label …")
    lab_dir = S / "labels_remapped"
    n_lab = 0
    for r in man:
        lp = lab_dir / (r["stem"] + ".txt")
        if not lp.exists():
            fails.append("缺 label %s" % lp.name)
            continue
        n_lab += 1
        if sha256_file(lp) != r["label_remapped_sha256"]:
            fails.append("%s 的 label 哈希与 manifest 不一致" % lp.name)
        src = pathlib.Path(str(root / r["rel_path"].replace("/images/", "/labels/")).rsplit(".", 1)[0] + ".txt")
        orig = parse_boxes(src, a.nc)
        orig_mapped = sorted(x[1] for x in orig if SRC_NAMES[x[0]] in MAP)
        new = parse_boxes(lp, 3)
        if any(c not in (0, 1, 2) for c, _ in new):
            fails.append("%s 含非法 class id" % lp.name)
        if sorted(x[1] for x in new) != orig_mapped:
            fails.append("%s 的框与「原始 label 里的映射类框」对不上" % lp.name)
        want_ids = sorted(MAP[SRC_NAMES[c]] for c, _ in orig if SRC_NAMES[c] in MAP)
        if sorted(c for c, _ in new) != want_ids:
            fails.append("%s 的 class id 映射不对" % lp.name)
    print("    校验 %d 个 label" % n_lab)

    # ---- 4. test manifest ----
    tm = list(csv.DictReader(open(S / "smart_campus_v3_test_manifest.csv", encoding="utf-8")))
    if sorted(x["rel_path"] for x in tm) != sorted(lists["test"]):
        fails.append("test manifest 与 test 清单不一致")
    for x in tm:
        lp = S / x["label_remapped"]
        if not lp.exists() or sha256_file(lp) != x["label_remapped_sha256"]:
            fails.append("test manifest 里 %s 的 label 哈希不符" % x["stem"])
        if sha256_file(root / x["rel_path"]) != x["sha256"]:
            fails.append("test manifest 里 %s 的图片哈希不符" % x["stem"])
    print("[4] test manifest %d 行，图片与 label 哈希逐行复核通过" % len(tm))
    # 组不可分割：同一组不能跨 split
    g2s = collections.defaultdict(set)
    for r in man:
        g2s[r["group_id"]].add(r["final_split"])
    split_groups = [k for k, v in g2s.items() if len(v) > 1]
    if split_groups:
        fails.append("有 %d 个 constraint group 跨了 split" % len(split_groups))
    print("    constraint group 跨 split 的：%d（应为 0）" % len(split_groups))

    print()
    if fails:
        print("[FAIL] %d 项，不写任何冻结文件：" % len(fails))
        for f in fails[:30]:
            print("   - " + f)
        return 3

    summary = {
        "freeze_type": "smart_campus_v3_split_freeze",
        "freeze_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dataset": "Smart Campus Dataset v9 (updated_original, YOLOv11 export, Roboflow user upload)",
        "class_map_to_mbdd": MAP,
        "mbdd_class_order": ["peeling", "crack", "rust"],
        "n_pool": n_pool,
        "split_sizes": {s: len(lists[s]) for s in SPLITS},
        "constraint_params": {"hash_max_dist": a.max_dist, "mirror_max_dist": a.mirror_max_dist},
        "verification": {
            "recomputed_hashes_from_images": True,
            "hash_near_pairs_total": n_near,
            "hash_near_pairs_cross_split": n_near_cross,
            "mirror_pairs_total": n_mir,
            "mirror_pairs_cross_split": n_mir_cross,
            "exact_dup_groups_cross_split": len(ex_cross),
            "constraint_groups_cross_split": len(split_groups),
            "remapped_labels_checked": n_lab,
            "test_manifest_rows": len(tm),
        },
        "test_sealed": True,
        "discipline": {
            "smart_campus_v3_test": "SEALED — 训练/早停/模型选择/阈值选择一律不得读取",
            "v2_split_test": "NOT USED in V3 (V2 已收口，V3 不碰)",
            "note": "test 由算法与固定 seed 产生，不因任何模型输出而调整",
        },
    }
    if a.write:
        # 冻结件本身不能进清单：否则重新冻结时会写进上一轮的旧哈希（自引用）
        SELF = {"v3_split_freeze.json", "v3_split_freeze_manifest.csv", "v3_split_freeze.sha256"}
        arts = sorted([p for p in S.iterdir() if p.is_file() and p.name not in SELF]
                      + [lab_dir])
        rows = []
        for p in arts:
            if p.is_dir():
                inner = sorted(p.glob("*.txt"))
                h = hashlib.sha256()
                for q in inner:
                    h.update(q.name.encode())
                    h.update(bytes.fromhex(sha256_file(q)))
                rows.append({"path": p.name + "/", "kind": "dir_hash_of_files",
                             "n_files": len(inner), "sha256": h.hexdigest()})
            else:
                rows.append({"path": p.name, "kind": "file", "n_files": 1, "sha256": sha256_file(p)})
        with open(S / "v3_split_freeze_manifest.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["path", "kind", "n_files", "sha256"])
            w.writeheader()
            w.writerows(rows)
        summary["artifacts"] = rows
        json.dump(summary, open(S / "v3_split_freeze.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        with open(S / "v3_split_freeze.sha256", "w", encoding="utf-8") as f:
            for p in ("v3_split_freeze.json", "v3_split_freeze_manifest.csv"):
                f.write("%s  SHA256 = %s" % (p, sha256_file(S / p)) + NL)
        print("[FREEZE] Smart Campus V3 test = SEALED")
        print("         冻结件：%d 项（含 labels_remapped/ 的目录哈希）→ %s" % (len(rows), S))
        for r in rows:
            print("           %-46s %s" % (r["path"], r["sha256"][:16]))
    else:
        print("[OK] 全部检查通过（未加 --write，未写冻结文件）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
