"""对 V3-CAMPUS-ADAPT 的**每个 epoch checkpoint** 在两个开发集上跑 post-hoc 评价（单进程内完成）。

为什么写成单进程：每个 checkpoint 调一次子进程会重复付模型加载 + AMP 检查的开销
（62 次调用 ≈ 十几分钟纯浪费）。这里只加载一次 ultralytics，逐个换权重。

命名注意：Ultralytics 在 `save_period=1` 下写出的是 **epoch0.pt … epoch29.pt**
（对应第 1…30 个 epoch），所以这里按数字排序枚举 `epoch*.pt`，不要假定从 1 开始。

对每个 checkpoint：
  A = MBDD split_v2 val (2,833)  → mAP 模式
  B = Smart Campus V3 val (329)  → mAP 模式
结果 JSON 的 schema 与 `eval_v3_checkpoints.py --mode map` 一致，供
`select_v3_checkpoint.py` 按预登记规则读取。**不读任何 test 集。**

用法：
  ... eval_all_checkpoints.py --weights-dir experiments/runs_v3/V3-CAMPUS-ADAPT/weights \
      --out-dir docs/evidence/v3_domain_adaptation/v3_training/per_epoch_eval
"""
import argparse
import hashlib
import json
import pathlib
import re
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[4]
SETS = {
    "mbdd_val": (ROOT / "data/splits_v2/data_orig.yaml", "val"),
    "sc_val": (ROOT / "data/v3_external/v3_sc_val.yaml", "val"),
}
EP_RE = re.compile(r"^epoch(\d+)\.pt$")
# 进度行格式：自检与正式打印共用同一份，避免"改了参数没改格式串"这类错误
PROGRESS_FMT = "[%2d/%d] %-12s A(MBDD) mAP50=%.4f | B(SC) mAP50=%.4f%s   [%.0fs]"


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
    ap.add_argument("--weights-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--alias-out-dir", default=None,
                    help="best.pt/last.pt 的**一致性核对**结果单独写到这里（不参与 selection）")
    a = ap.parse_args()

    W = pathlib.Path(a.weights_dir).resolve()
    O = pathlib.Path(a.out_dir).resolve()
    O.mkdir(parents=True, exist_ok=True)

    cks = sorted((p for p in W.glob("epoch*.pt") if EP_RE.match(p.name)),
                 key=lambda p: int(EP_RE.match(p.name).group(1)))
    for extra in ("best.pt", "last.pt"):
        if (W / extra).exists():
            cks.append(W / extra)
    print("[info] 待评价 checkpoint %d 个：%s …" % (len(cks), ", ".join(c.stem for c in cks[:3])))
    sys.stdout.flush()

    # 自检：用假值先把进度行走一遍。格式化串与参数不匹配这类错误，
    # 必须在花掉几十分钟之前就暴露（上一次就是这样白跑了一轮）。
    print(PROGRESS_FMT % (0, len(cks), "selftest", 0.0, 0.0, "", 0.0))
    sys.stdout.flush()

    from ultralytics import YOLO
    t0 = time.time()
    rows = []
    for i, ck in enumerate(cks, 1):
        # Ultralytics 的 epochN.pt 里 N 从 0 起 → 人类可读的 epoch 号是 N+1
        m = EP_RE.match(ck.name)
        human_epoch = int(m.group(1)) + 1 if m else ck.stem
        is_alias = m is None
        rec = {"checkpoint": str(ck), "checkpoint_sha256": sha256_file(ck),
               "weights_name": ck.name, "epoch": human_epoch, "is_alias": is_alias}
        model = YOLO(str(ck))
        for setname, (yaml, split) in SETS.items():
            if is_alias and a.alias_out_dir:
                AO = pathlib.Path(a.alias_out_dir); AO.mkdir(parents=True, exist_ok=True)
                out = AO / ("alias_%s.%s.json" % (ck.stem, setname))
            else:
                out = O / ("epoch%s_%s_map.json" % (human_epoch, setname))
            if out.exists():
                d = json.load(open(out, encoding="utf-8"))
            else:
                v = model.val(data=str(yaml), split=split, imgsz=640, batch=a.batch, device=0,
                              conf=0.001, iou=0.7, max_det=300, plots=False, verbose=False,
                              project=str(ROOT / "experiments/runs_v3/_posthoc"),
                              name="%s_%s" % (ck.stem, setname), exist_ok=True)
                names = v.names if isinstance(v.names, dict) else {i2: n for i2, n in enumerate(v.names)}
                per_class = []
                for cid, cname in names.items():
                    try:
                        ap50 = float(v.box.ap50[cid])
                    except Exception:
                        ap50 = None
                    try:
                        ap = float(v.box.ap[cid])
                    except Exception:
                        ap = None
                    per_class.append({"class_id": int(cid), "class_name": cname,
                                      "AP50": ap50, "AP50-95": ap})
                d = {"checkpoint": str(ck), "checkpoint_sha256": rec["checkpoint_sha256"],
                     "set": setname, "mode": "map",
                     "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                     "P": float(v.box.mp), "R": float(v.box.mr),
                     "mAP50": float(v.box.map50), "mAP50-95": float(v.box.map),
                     "per_class": per_class}
                json.dump(d, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            rec[setname] = {"mAP50": d["mAP50"], "mAP50_95": d["mAP50-95"]}
        if not is_alias:
            rows.append(rec)
        print(PROGRESS_FMT % (i, len(cks), ck.name, rec["mbdd_val"]["mAP50"],
                              rec["sc_val"]["mAP50"],
                              "  (alias, 不入候选)" if is_alias else "", time.time() - t0))
        sys.stdout.flush()

    json.dump(rows, open(O / "all_checkpoints_summary.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("[done] 总耗时 %.1f 分钟；汇总写入 %s" % ((time.time() - t0) / 60, O / "all_checkpoints_summary.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
