"""外部数据入库（intake）：把用户手工下载的 zip 安全解开，并记录来源证据。

它会做的事：
  1. 记下 zip 的文件名、字节数、SHA256、mtime（**这是「实际下载版本」的唯一凭据**）；
  2. 检查压缩包内每个条目的路径，拒绝绝对路径 / `..` 穿越 / 盘符路径（下载物是不可信数据）；
  3. 解到**一个必须为空的新目录**（默认拒绝覆盖，除非显式 --allow-nonempty）；
  4. 打印顶层结构，定位 data.yaml / README.roboflow.txt 之类的来源文件；
  5. 若有 `README.roboflow.txt`（Roboflow 导出自带，含项目链接与许可证），
     原样复制一份到证据目录，作为「来源 / 许可证」的一手凭据；
  6. 写 `intake_record.json`（结构化）与 `intake_record.md`（人读）。

**不执行压缩包里的任何东西**，只解压与读取文本。

用法：
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe \
    docs/evidence/v3_domain_adaptation/_scripts/intake_dataset.py \
    --zip data/v3_external/smart_campus/_incoming/<name>.zip \
    --dest data/v3_external/smart_campus/raw/<name> \
    --evidence docs/evidence/v3_domain_adaptation/smart_campus/provenance
"""
import argparse
import hashlib
import json
import pathlib
import shutil
import sys
import time
import zipfile

TEXT_KEEP = ("readme", "license", "licence", "data.yaml", "data.yml", "dataset.yaml", "citation")


def sha256_file(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def unsafe_reason(name):
    """返回不安全原因，安全则返回 None。"""
    if not name or name.endswith("/"):
        return None                                   # 目录条目本身无所谓
    p = pathlib.PurePosixPath(name.replace("\\", "/"))
    if p.is_absolute():
        return "绝对路径"
    if any(part == ".." for part in p.parts):
        return "包含 .. 的路径穿越"
    if len(str(p).split(":")[0]) == 1 and ":" in str(p)[:3]:
        return "带盘符的路径"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--evidence", required=True, help="来源证据落地目录")
    ap.add_argument("--allow-nonempty", action="store_true")
    a = ap.parse_args()

    zp = pathlib.Path(a.zip).resolve()
    dest = pathlib.Path(a.dest).resolve()
    ev = pathlib.Path(a.evidence).resolve()
    if not zp.is_file():
        print("[FATAL] 找不到 zip：%s" % zp)
        return 2
    if dest.exists() and any(dest.iterdir()) and not a.allow_nonempty:
        print("[FATAL] 目标目录非空：%s（拒绝覆盖；确需则加 --allow-nonempty）" % dest)
        return 2
    dest.mkdir(parents=True, exist_ok=True)
    ev.mkdir(parents=True, exist_ok=True)

    zsha = sha256_file(zp)
    rec = {
        "zip_path": str(zp), "zip_name": zp.name, "zip_bytes": zp.stat().st_size,
        "zip_sha256": zsha, "zip_mtime_local": time.strftime(
            "%Y-%m-%dT%H:%M:%S%z", time.localtime(zp.stat().st_mtime)),
        "intake_time_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dest": str(dest),
    }
    print("[info] zip %s | %d bytes | sha256 %s" % (zp.name, rec["zip_bytes"], zsha[:16]))

    with zipfile.ZipFile(zp) as z:
        names = z.namelist()
        bad = [(n, unsafe_reason(n)) for n in names if unsafe_reason(n)]
        rec["n_entries"] = len(names)
        rec["unsafe_entries"] = bad
        if bad:
            print("[FATAL] 压缩包内有 %d 个不安全路径，拒绝解压：" % len(bad))
            for n, r in bad[:10]:
                print("   - %s (%s)" % (n, r))
            return 3
        z.extractall(dest)
    print("[info] 解压完成 → %s（%d 个条目）" % (dest, rec["n_entries"]))

    # 顶层结构
    top = sorted({p.name + ("/" if p.is_dir() else "") for p in dest.iterdir()})
    rec["top_level"] = top
    print("[info] 顶层：%s" % top)

    # 找来源/许可类文本文件（一律原样留档，不改写）
    kept = []
    for p in sorted(dest.rglob("*")):
        if p.is_file() and any(k in p.name.lower() for k in TEXT_KEEP) and p.stat().st_size < 2 << 20:
            tgt = ev / p.name
            if tgt.exists():
                tgt = ev / (p.parent.name + "__" + p.name)
            shutil.copy2(p, tgt)
            kept.append({"source_in_archive": str(p.relative_to(dest)).replace("\\", "/"),
                         "kept_as": tgt.name, "bytes": p.stat().st_size,
                         "sha256": sha256_file(p)})
    rec["provenance_files_kept"] = kept
    print("[info] 留档来源/许可文本 %d 个 → %s" % (len(kept), ev))
    for k in kept:
        print("   - %s" % k["source_in_archive"])

    # 是否像 YOLO 检测导出
    dy = [p for p in dest.rglob("data.y*ml")] + [p for p in dest.rglob("dataset.y*ml")]
    rec["data_yaml_candidates"] = [str(p.relative_to(dest)).replace("\\", "/") for p in dy]
    splits = {sp: sum(1 for _ in (dest / sp / "images").glob("*"))
              for sp in ("train", "valid", "test") if (dest / sp / "images").is_dir()}
    rec["images_per_split_dir"] = splits
    rec["looks_like_yolo_export"] = bool(dy) and bool(splits)
    print("[info] data.yaml: %s | 各 split 图片数: %s | 像 YOLO 导出: %s"
          % (rec["data_yaml_candidates"], splits, rec["looks_like_yolo_export"]))

    (ev / "intake_record.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    md = ["# 数据入库记录", "",
          "| 项 | 值 |", "|---|---|",
          "| zip | `%s` |" % rec["zip_name"],
          "| 字节数 | %d |" % rec["zip_bytes"],
          "| SHA256 | `%s` |" % rec["zip_sha256"],
          "| zip mtime | `%s` |" % rec["zip_mtime_local"],
          "| 入库时间 | `%s` |" % rec["intake_time_local"],
          "| 解压到 | `%s` |" % rec["dest"],
          "| 条目数 | %d |" % rec["n_entries"],
          "| 顶层 | %s |" % ", ".join(top),
          "| 像 YOLO 检测导出 | %s |" % rec["looks_like_yolo_export"], "",
          "## 留档的来源/许可文本", ""]
    md += ["- `%s` → `%s`" % (k["source_in_archive"], k["kept_as"]) for k in kept] or ["- （无）"]
    md += ["", "> zip 的 SHA256 是「实际下载版本」的唯一凭据。",
           "> 解压未执行压缩包内任何内容；不安全路径（绝对路径 / `..` / 盘符）会被拒绝。"]
    (ev / "intake_record.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("[OK] 入库记录写到 %s" % ev)
    return 0


if __name__ == "__main__":
    sys.exit(main())
