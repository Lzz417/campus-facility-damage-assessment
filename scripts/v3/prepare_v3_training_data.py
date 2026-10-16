"""把冻结的 V3 划分物化成可训练的 YOLO 目录，并生成训练用的 data yaml 与合并训练清单。

为什么需要这一步：Smart Campus 的 V3 划分是**跨原始目录重新分配**的
（原 train/valid/test 三个目录被打散），而且 label 是统一放在 `labels_remapped/` 的扁平目录里。
Ultralytics 需要「images/<split>/*.jpg 与 labels/<split>/*.txt 一一对应」的结构，
所以这里把池内的图与重映射 label 复制成规范布局（**只复制，不改原件**）。

产物：
  data/v3_external/smart_campus/v3_dataset/{images,labels}/{train,val,test}/
  data/v3_external/v3_campus_adapt.yaml          # train = MBDD train_orig + V3 train；val = MBDD val
  docs/evidence/v3_domain_adaptation/v3_training/
      combined_train_manifest.csv                # 两来源逐文件 sha256
      combined_train_manifest.sha256
      prepare_record.json

用法：
  ... prepare_v3_training_data.py --write
"""
import argparse
import csv
import hashlib
import json
import pathlib
import shutil
import sys
import time

NL = chr(10)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--split-dir",
                    default="docs/evidence/v3_domain_adaptation/smart_campus/v3_split")
    ap.add_argument("--smart-campus-raw",
                    default="data/v3_external/smart_campus/raw/v9-updated_original.yolov11")
    ap.add_argument("--out-dataset", default="data/v3_external/smart_campus/v3_dataset")
    ap.add_argument("--yaml-out", default="data/v3_external/v3_campus_adapt.yaml")
    ap.add_argument("--evidence", default="docs/evidence/v3_domain_adaptation/v3_training")
    ap.add_argument("--mbdd-train", default="data/splits_v2/images/train_orig")
    ap.add_argument("--mbdd-val", default="data/splits_v2/images/val")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    R = pathlib.Path(a.root).resolve()
    S = R / a.split_dir
    raw = R / a.smart_campus_raw
    ds = R / a.out_dataset
    ev = R / a.evidence
    man = list(csv.DictReader(open(S / "smart_campus_v3_split_manifest.csv", encoding="utf-8")))
    lab_src = S / "labels_remapped"
    print("[info] V3 划分 %d 张；来源图 %s" % (len(man), raw))

    if not a.write:
        for s in SPLITS:
            n = sum(1 for r in man if r["final_split"] == s)
            print("   %-5s %d 张（dry-run 未落盘）" % (s, n))
        return 0

    ev.mkdir(parents=True, exist_ok=True)
    rec = {"prepared_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "split_dir": str(S), "dataset_out": str(ds), "splits": {}}
    rows = []
    for s in SPLITS:
        (ds / "images" / s).mkdir(parents=True, exist_ok=True)
        (ds / "labels" / s).mkdir(parents=True, exist_ok=True)
        n = 0
        for r in man:
            if r["final_split"] != s:
                continue
            src_img = raw / r["rel_path"]
            dst_img = ds / "images" / s / src_img.name
            shutil.copy2(src_img, dst_img)
            if sha256_file(dst_img) != r["sha256"]:
                print("[FATAL] 复制后哈希不符：%s" % dst_img)
                return 3
            src_lab = lab_src / (r["stem"] + ".txt")
            dst_lab = ds / "labels" / s / (r["stem"] + ".txt")
            shutil.copy2(src_lab, dst_lab)
            if sha256_file(dst_lab) != r["label_remapped_sha256"]:
                print("[FATAL] 复制后 label 哈希不符：%s" % dst_lab)
                return 3
            rows.append({"source": "smart_campus_v3", "split": s, "file": str(dst_img.relative_to(R)),
                         "image_sha256": r["sha256"], "label_sha256": r["label_remapped_sha256"]})
            n += 1
        rec["splits"][s] = n
        print("   %-5s 复制 %d 张（图 + label，哈希已复核）" % (s, n))

    # MBDD 训练图（只登记哈希，不复制：原目录结构本来就是规范的）
    mb_train = R / a.mbdd_train
    mb_val = R / a.mbdd_val
    n_mb = 0
    for p in sorted(mb_train.iterdir()):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        rows.append({"source": "mbdd_split_v2_train_orig", "split": "train",
                     "file": str(p.relative_to(R)), "image_sha256": sha256_file(p),
                     "label_sha256": ""})
        n_mb += 1
    rec["mbdd_train_images"] = n_mb
    print("   MBDD train_orig 登记 %d 张" % n_mb)

    # 合并训练清单
    rows.sort(key=lambda r: (r["source"], r["file"]))
    cm = ev / "combined_train_manifest.csv"
    with open(cm, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["source", "split", "file", "image_sha256", "label_sha256"])
        w.writeheader()
        w.writerows(rows)
    h = sha256_file(cm)
    (ev / "combined_train_manifest.sha256").write_text(
        "combined_train_manifest.csv  SHA256 = %s%s" % (h, NL), encoding="utf-8")
    rec["combined_train_manifest_sha256"] = h
    rec["combined_train_count"] = n_mb + rec["splits"]["train"]
    print("[写] 合并训练清单 %d 行（MBDD %d + V3 train %d）→ %s"
          % (len(rows), n_mb, rec["splits"]["train"], h[:16]))

    # 训练用 data yaml
    yml = ("# V3-CAMPUS-ADAPT 训练配置" + NL +
           "# train = MBDD split_v2 train originals (7752) + Smart Campus V3 train (%d)" % rec["splits"]["train"] + NL +
           "# val  = MBDD split_v2 val (2833) —— 仅作训练期内部记账；" + NL +
           "#        正式指标一律来自 post-hoc evaluator（见 v3_model_selection_protocol.md）" + NL +
           "# 禁止：split_v2 val/test 不得训练；Smart Campus V3 val/test 不得训练。" + NL +
           "# class 顺序与 MBDD 一致：0 peeling / 1 crack / 2 rust" + NL +
           "path: %s" % (R / "data/v3_external").as_posix() + NL +
           "train:" + NL +
           "  - %s" % mb_train.as_posix() + NL +
           "  - %s" % (ds / "images" / "train").as_posix() + NL +
           "val: %s" % mb_val.as_posix() + NL +
           "names:" + NL +
           "  0: peeling" + NL + "  1: crack" + NL + "  2: rust" + NL)
    (R / a.yaml_out).write_text(yml, encoding="utf-8")
    rec["yaml"] = a.yaml_out
    rec["yaml_sha256"] = sha256_file(R / a.yaml_out)
    print("[写] %s (%s)" % (a.yaml_out, rec["yaml_sha256"][:16]))
    json.dump(rec, open(ev / "prepare_record.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
