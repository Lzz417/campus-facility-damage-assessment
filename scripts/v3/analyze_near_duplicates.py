"""近似重复的二次判定：光有 dHash/pHash 距离不足以断定「重复」。

为什么要这一步：在 Smart Campus v9 上，`min(dHash,pHash) <= 2` 命中 **805 对**，
但 dHash 只有 4231/4323 个不同取值——说明不是哈希退化，而是**哈希空间高度聚集**。
dHash 的判据是 9x8 降采样后的「相邻像素大小关系」，对**低纹理**图像
（大片路面、墙面、天空）会给出几乎相同的符号图案，于是**视觉上毫不相干的两张图也能量到 0~2**。
因此「距离 ≤2」在本数据集上**不能**单独作为重复证据。

本脚本对候选对再做一次**直接像素级复核**：
  * 把两图各降到 32x32 灰度，做 **零均值单位方差归一化**，
    算**归一化相关系数** corr（±1，1 = 完全同图）；
  * 同时给 32x32 的**归一化平均绝对差** mad（越小越像）。
判读门槛（写死在这里，事后不改）：
  * `corr >= 0.98` → **strong**：几乎可以肯定是同一张图的重编码/缩放；
  * `0.90 <= corr < 0.98` → **moderate**：同场景/同一缺陷的不同拍摄，需人工看；
  * `corr < 0.90` → **weak**：哈希碰撞，**不是**重复。

同时输出**成对 contact sheet**，让人（含我自己）能直接看，而不是只看数字。

用法：
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe \
    docs/evidence/v3_domain_adaptation/_scripts/analyze_near_duplicates.py \
    --dataset-root data/v3_external/smart_campus/raw/v9-updated_original.yolov11 \
    --images-manifest docs/evidence/v3_domain_adaptation/smart_campus/audit/smart_campus_images_manifest.csv \
    --out docs/evidence/v3_domain_adaptation/smart_campus/near_dup
"""
import argparse
import collections
import csv
import pathlib
import sys

import cv2
import numpy as np

STRONG = 0.98
MODERATE = 0.90


def load_gray32(root, rel):
    p = root / rel
    im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None
    return cv2.resize(im, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)


def corr_mad(a, b):
    """零均值单位方差归一化后的相关系数与平均绝对差。"""
    a = a - a.mean()
    b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return 0.0, 255.0
    c = float((a * b).sum() / (na * nb))
    sa, sb = a.std() + 1e-6, b.std() + 1e-6
    mad = float(np.abs(a / sa - b / sb).mean())
    return c, mad


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


def pair_tile(root, ra, rb, dist_min, corr, mad, same):
    def thumb(rel):
        im = cv2.imread(str(root / rel), cv2.IMREAD_COLOR)
        if im is None:
            return np.full((150, 200, 3), 40, np.uint8)
        return cv2.resize(im, (200, 150), interpolation=cv2.INTER_AREA)
    A, B = thumb(ra["rel_path"]), thumb(rb["rel_path"])
    tile = np.full((150 + 40, 200 + 4 + 200, 3), 24, np.uint8)
    tile[0:150, 0:200] = A
    tile[0:150, 204:404] = B
    col = (0, 200, 0) if corr >= STRONG else ((0, 200, 255) if corr >= MODERATE else (0, 0, 255))
    txt = "d=%d corr=%.3f %s %s" % (dist_min, corr,
                                    "SAME-split" if same else "CROSS-split",
                                    "STRONG" if corr >= STRONG else ("MODERATE" if corr >= MODERATE else "weak"))
    cv2.putText(tile, txt, (6, 172), cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1, cv2.LINE_AA)
    cv2.putText(tile, ra["stem"][:34], (6, 186), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(tile, rb["stem"][:34], (206, 186), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    return tile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--images-manifest", required=True)
    ap.add_argument("--max-dist", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs-per-sheet", type=int, default=9)
    ap.add_argument("--sheets", type=int, default=6)
    ap.add_argument("--stratify", action="store_true",
                    help="按 STRONG/MODERATE/weak 各出一屏，用于验证门槛本身")
    ap.add_argument("--check-mirror", action="store_true",
                    help="额外检查水平镜像副本（dHash/pHash 对镜像图不敏感，会漏）")
    a = ap.parse_args()

    root = pathlib.Path(a.dataset_root).resolve()
    out = pathlib.Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(a.images_manifest, encoding="utf-8")))
    print("[info] 载入逐图清单 %d 行" % len(rows))

    dh = np.array([int(r["dhash"], 16) for r in rows], dtype=np.uint64)
    ph = np.array([int(r["phash"], 16) for r in rows], dtype=np.uint64)
    pd = pairs_at(dh, a.max_dist)
    pp = pairs_at(ph, a.max_dist)
    best = {}
    for (i, j, d) in pd:
        best[(i, j)] = [d, 99]
    for (i, j, d) in pp:
        k = (i, j)
        if k in best:
            best[k][1] = d
        else:
            best[k] = [99, d]
    cand = {k: v for k, v in best.items() if min(v) <= a.max_dist}
    print("[info] dHash 命中 %d 对 / pHash 命中 %d 对 / 取 min 后候选 %d 对"
          % (len(pd), len(pp), len(cand)))

    # 像素级复核
    cache = {}

    def g32(rel):
        if rel not in cache:
            cache[rel] = load_gray32(root, rel)
        return cache[rel]

    # ---- 可选：镜像副本检查（dHash 比较的是相邻像素的左右关系，镜像后会变；
    #      pHash 的 DCT 幅值对镜像也不对称 —— 两者都可能漏掉水平翻转副本）
    mirror_pairs = []
    if a.check_mirror:
        def hashes_from_g32(g):
            g8 = g.astype(np.uint8)
            g9 = cv2.resize(g8, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int32)
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

        fl_d, fl_p = [], []
        for r in rows:
            g = g32(r["rel_path"])
            if g is None:
                fl_d.append(np.uint64(0))
                fl_p.append(np.uint64(0))
                continue
            fd, fp = hashes_from_g32(cv2.flip(g, 1))
            fl_d.append(fd)
            fl_p.append(fp)
        N = len(rows)
        all_d = np.concatenate([dh, np.array(fl_d, dtype=np.uint64)])
        all_p = np.concatenate([ph, np.array(fl_p, dtype=np.uint64)])
        md = pairs_at(all_d, a.max_dist)
        mp = pairs_at(all_p, a.max_dist)
        bestm = {}
        for (i, j, d) in md:
            bestm[(i, j)] = [d, 99]
        for (i, j, d) in mp:
            if (i, j) in bestm:
                bestm[(i, j)][1] = d
            else:
                bestm[(i, j)] = [99, d]
        for (i, j), v in sorted(bestm.items()):
            if min(v) > a.max_dist:
                continue
            # 只保留「一个来自正常半、一个来自翻转半」的组合
            if (i < N) == (j < N):
                continue
            ni, nj = (i, j - N) if i < N else (j, i - N)
            A, B = g32(rows[ni]["rel_path"]), g32(rows[nj]["rel_path"])
            if A is None or B is None:
                continue
            c, m = corr_mad(cv2.flip(A, 1), B)
            mirror_pairs.append({
                "a_stem": rows[ni]["stem"], "b_stem": rows[nj]["stem"],
                "a_split": rows[ni]["split"], "b_split": rows[nj]["split"],
                "same_split": int(rows[ni]["split"] == rows[nj]["split"]),
                "min_hash_dist": min(v), "corr32_flipped_vs_other": round(c, 5),
                "verdict": "STRONG" if c >= STRONG else ("MODERATE" if c >= MODERATE else "weak"),
            })
        mirror_pairs.sort(key=lambda r: -r["corr32_flipped_vs_other"])
        with open(out / "mirror_pairs.csv", "w", newline="", encoding="utf-8") as f:
            if mirror_pairs:
                w = csv.DictWriter(f, fieldnames=list(mirror_pairs[0].keys()))
                w.writeheader()
                w.writerows(mirror_pairs)
        mv = collections.Counter(r["verdict"] for r in mirror_pairs)
        print("[info] 镜像候选 %d 对：%s（跨 split %d 对）"
              % (len(mirror_pairs), dict(mv),
                 sum(1 for r in mirror_pairs if not r["same_split"])))
        # 让「镜像也是重复」这句话可被目视复核：左=原图，右=镜像后的另一张
        top_m = mirror_pairs[:9]
        if top_m:
            mrow = {r["stem"]: r for r in rows}
            cols2, rows2, cw2, ch2 = 3, 3, 408, 190
            sheet = np.full((rows2 * ch2 + 34, cols2 * cw2, 3), 24, np.uint8)
            cv2.putText(sheet, "MIRROR check | left = image A, right = image B (A flipped "
                              "horizontally to compare) | STRONG first", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            for k, r in enumerate(top_m):
                rr, cc = divmod(k, cols2)
                y, x = 34 + rr * ch2, cc * cw2
                t = pair_tile(root, mrow[r["a_stem"]], mrow[r["b_stem"]],
                              r["min_hash_dist"], r["corr32_flipped_vs_other"], 0.0,
                              r["same_split"])
                sheet[y:y + t.shape[0], x:x + t.shape[1]] = t[:ch2, :cw2]
            cv2.imwrite(str(out / "mirror_sheet_01.jpg"), sheet,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])

    out_rows = []
    for (i, j), (dd, dpp) in sorted(cand.items()):
        ra, rb = rows[i], rows[j]
        A, B = g32(ra["rel_path"]), g32(rb["rel_path"])
        if A is None or B is None:
            c, m = 0.0, 255.0
        else:
            c, m = corr_mad(A, B)
        out_rows.append({
            "a_stem": ra["stem"], "b_stem": rb["stem"],
            "a_split": ra["split"], "b_split": rb["split"],
            "same_split": int(ra["split"] == rb["split"]),
            "min_hash_dist": min(dd, dpp), "dhash_dist": dd, "phash_dist": dpp,
            "corr32": round(c, 5), "mad32": round(m, 5),
            "verdict": "STRONG" if c >= STRONG else ("MODERATE" if c >= MODERATE else "weak"),
            "a_rel": ra["rel_path"], "b_rel": rb["rel_path"],
        })
    out_rows.sort(key=lambda r: -r["corr32"])

    with open(out / "near_dup_pairs_reviewed.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()) if out_rows else ["a_stem"])
        w.writeheader()
        w.writerows(out_rows)

    verdict = collections.Counter(r["verdict"] for r in out_rows)
    cross = collections.Counter((r["verdict"], "cross" if not r["same_split"] else "same")
                                for r in out_rows)
    n_cross_all = sum(1 for r in out_rows if not r["same_split"])
    print("[info] 像素级复核：%s" % dict(verdict))
    print("[info] 跨 split 候选 %d 对；其中 STRONG %d 对、MODERATE %d 对"
          % (n_cross_all, cross[("STRONG", "cross")], cross[("MODERATE", "cross")]))

    # 分组（仍按哈希候选合并，供划分 builder 参考；是否采用 STRONG-only 由复核结论决定）
    uf_all, uf_strong = UF(len(rows)), UF(len(rows))
    for (i, j), (dd, dpp) in cand.items():
        uf_all.union(i, j)
    for r in out_rows:
        if r["verdict"] == "STRONG":
            i = next(k for k, x in enumerate(rows) if x["stem"] == r["a_stem"])
            j = next(k for k, x in enumerate(rows) if x["stem"] == r["b_stem"])
            uf_strong.union(i, j)

    def groups(uf):
        g = collections.defaultdict(list)
        for i in range(len(rows)):
            g[uf.find(i)].append(i)
        return sorted([v for v in g.values() if len(v) > 1], key=len, reverse=True)

    g_all, g_strong = groups(uf_all), groups(uf_strong)
    print("[info] 组：按哈希候选 %d 组（最大 %d）／按 STRONG 复核 %d 组（最大 %d）"
          % (len(g_all), len(g_all[0]) if g_all else 0, len(g_strong), len(g_strong[0]) if g_strong else 0))
    with open(out / "near_dup_groups.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["basis", "group_id", "size", "members_stems", "members_splits"])
        for basis, gs in (("hash_candidate", g_all), ("strong_only", g_strong)):
            for gi, mem in enumerate(gs):
                w.writerow([basis, gi, len(mem),
                            "|".join(rows[m]["stem"] for m in mem),
                            "|".join(rows[m]["split"] for m in mem)])

    # 成对 contact sheet
    stem2row = {r["stem"]: r for r in rows}
    rng = np.random.default_rng(0)
    if a.stratify:
        # 按判定分层各抽一屏，用来「验证门槛本身」而不是只看最像的那批
        sheets_sel = []
        for v in ("STRONG", "MODERATE", "weak"):
            pool = [r for r in out_rows if r["verdict"] == v]
            pick = [pool[i] for i in sorted(rng.choice(len(pool),
                                                       size=min(len(pool), a.pairs_per_sheet),
                                                       replace=False))] if pool else []
            sheets_sel.append((v, pick))
    else:
        n = a.pairs_per_sheet * a.sheets
        top = out_rows[:n // 2]
        rest = out_rows[len(top):]
        rnd = [rest[i] for i in sorted(rng.choice(len(rest), size=min(len(rest), n - len(top)),
                                                  replace=False))] if rest else []
        sel = top + rnd
        sheets_sel = [(None, sel[s * a.pairs_per_sheet:(s + 1) * a.pairs_per_sheet])
                      for s in range(a.sheets)]
    made = 0
    for si, (vtag, chunk) in enumerate(sheets_sel):
        s = si
        if not chunk:
            continue
        cols, rowsn = 3, 3
        cw, ch = 408, 190
        sheet = np.full((rowsn * ch + 34, cols * cw, 3), 24, np.uint8)
        head = ("near-duplicate pixel review | d=min(dHash,pHash) | corr=32x32 normalized "
                "| stratum: %s | sheet %d" % (vtag or "strongest-then-random", s + 1))
        cv2.putText(sheet, head, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        for k, r in enumerate(chunk):
            rr, cc = divmod(k, cols)
            y, x = 34 + rr * ch, cc * cw
            t = pair_tile(root, stem2row[r["a_stem"]], stem2row[r["b_stem"]],
                          r["min_hash_dist"], r["corr32"], r["mad32"], r["same_split"])
            sheet[y:y + t.shape[0], x:x + t.shape[1]] = t[:ch, :cw]
        fp = out / ("pairs_sheet_%s_%02d.jpg" % (vtag or "main", s + 1))
        cv2.imwrite(str(fp), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
        made += 1
    print("[OK] 写了 %d 张成对 sheet、pairs csv、groups csv → %s" % (made, out))

    L = ["# 近似重复二次判定（像素级复核）", "",
         "- 候选口径：`min(dHash,pHash) <= %d`" % a.max_dist,
         "- 候选对：**%d**（dHash 命中 %d，pHash 命中 %d）" % (len(cand), len(pd), len(pp)),
         "- 像素级判据：32x32 灰度、零均值单位方差归一化后的相关系数 corr", "",
         "| 判定 | 门槛 | 对数 | 其中跨 split |", "|---|---|---|---|"]
    for v in ("STRONG", "MODERATE", "weak"):
        L.append("| %s | %s | %d | %d |"
                 % (v, {"STRONG": "corr >= 0.98", "MODERATE": "0.90 <= corr < 0.98",
                        "weak": "corr < 0.90"}[v],
                    verdict[v], cross[(v, "cross")]))
    L += ["",
          "- 分组：按哈希候选 **%d** 组（最大 %d）；只按 STRONG 复核 **%d** 组（最大 %d）"
          % (len(g_all), len(g_all[0]) if g_all else 0, len(g_strong), len(g_strong[0]) if g_strong else 0),
          "- 成对 sheet：`pairs_sheet_*.jpg`（最强在前，其余随机抽样，seed=0）",
          "- 明细：`near_dup_pairs_reviewed.csv`、`near_dup_groups.csv`", "",
          "> 结论必须**看图**再下：哈希距离单独不足以判定重复（低纹理图像会聚集）。"]
    (out / "near_dup_review.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
