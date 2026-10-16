"""V3 的 post-hoc evaluator：对给定 checkpoint 在两个**开发集**上出正式指标。

开发集 A = MBDD split_v2 val (2,833)
开发集 B = Smart Campus V3 val (329)     ← Smart Campus V3 test 永不被读取

两种模式：
  mAP 模式  `--mode map`    ：ultralytics 标准口径 conf=0.001 / iou=0.7 / max_det=300
                              输出 mAP50、mAP50-95、逐类 AP50
  扫描模式  `--mode sweep`  ：在固定 conf 列表上统计 detection P/R/F1
                              （NMS iou=0.45、max_det=30、匹配 IoU=0.5，贪心匹配）

用法：
  ... eval_v3_checkpoints.py --ckpt <path> --set sc_val  --mode map  --out <json>
  ... eval_v3_checkpoints.py --ckpt <path> --set mbdd_val --mode map --out <json>
  ... eval_v3_checkpoints.py --ckpt <path> --set sc_val  --mode sweep --out <json>
"""
import argparse
import collections
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[4]
SETS = {
    "mbdd_val": {"yaml": ROOT / "data/splits_v2/data_orig.yaml", "split": "val",
                 "images": ROOT / "data/splits_v2/images/val", "n": 2833},
    "sc_val": {"yaml": ROOT / "data/v3_external/v3_sc_val.yaml", "split": "val",
               "images": ROOT / "data/v3_external/smart_campus/v3_dataset/images/val", "n": 329},
    # ⚠️ 只在正式 test 评价时使用，且必须先有 v3_test_unseal.json（由 run_v3_test_comparison.py 把关）
    "sc_test": {"yaml": ROOT / "data/v3_external/v3_sc_test.yaml", "split": "val",
                "images": ROOT / "data/v3_external/smart_campus/v3_dataset/images/test", "n": 329},
}
CONFS = [0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
NMS_IOU = 0.45
MAX_DET = 30
MATCH_IOU = 0.5
NAMES = ["peeling", "crack", "rust"]


def load_gt(img_path, nc=3):
    """读 YOLO label（归一化 xywh）→ 绝对坐标 xyxy 的数组。"""
    lp = pathlib.Path(str(img_path).replace("images", "labels", 1)).with_suffix(".txt")
    out = []
    if lp.exists():
        for line in lp.read_text(encoding="utf-8", errors="replace").splitlines():
            q = line.split()
            if len(q) >= 5:
                c = int(float(q[0]))
                if 0 <= c < nc:
                    out.append((c, *[float(x) for x in q[1:5]]))
    return out


def iou_xyxy(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def greedy_match(dets, gts, iou_thr):
    """按置信度降序贪心匹配同类框，返回 (tp, fp, fn)。"""
    dets = sorted(dets, key=lambda d: -d[0])
    used = [False] * len(gts)
    tp = 0
    for conf, cls, box in dets:
        best, bj = iou_thr, -1
        for j, (gc, gb) in enumerate(gts):
            if used[j] or gc != cls:
                continue
            v = iou_xyxy(box, gb)
            if v > best:
                best, bj = v, j
        if bj >= 0:
            used[bj] = True
            tp += 1
    return tp, len(dets) - tp, len(gts) - tp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--set", dest="setname", required=True, choices=list(SETS))
    ap.add_argument("--mode", required=True, choices=["map", "sweep"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--report-conf", type=float, default=None,
                    help="扫描模式：额外报告该阈值下的 mean boxes/image 与 no-prediction image rate")
    a = ap.parse_args()

    from ultralytics import YOLO
    cfg = SETS[a.setname]
    ck = pathlib.Path(a.ckpt).resolve()
    res = {"checkpoint": str(ck), "checkpoint_sha256": None, "set": a.setname,
           "mode": a.mode, "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    import hashlib
    res["checkpoint_sha256"] = hashlib.sha256(ck.read_bytes()).hexdigest()
    model = YOLO(str(ck))
    print("[eval] %s | %s | %s" % (ck.name, a.setname, a.mode))
    sys.stdout.flush()

    if a.mode == "map":
        m = model.val(data=str(cfg["yaml"]), split=cfg["split"], imgsz=640, batch=a.batch,
                      device=0, conf=0.001, iou=0.7, max_det=300, plots=False, verbose=False,
                      project=str(ROOT / "experiments/runs_v3/_posthoc"), name="%s_%s_map" % (ck.stem, a.setname),
                      exist_ok=True)
        names = m.names if isinstance(m.names, dict) else {i: n for i, n in enumerate(m.names)}
        per_class = []
        for i, cname in names.items():
            try:
                ap50 = float(m.box.ap50[i])
            except Exception:
                ap50 = None
            try:
                ap = float(m.box.ap[i])
            except Exception:
                ap = None
            per_class.append({"class_id": int(i), "class_name": cname, "AP50": ap50,
                              "AP50-95": ap})
        res.update({"P": float(m.box.mp), "R": float(m.box.mr),
                    "mAP50": float(m.box.map50), "mAP50-95": float(m.box.map),
                    "per_class": per_class})
        print("[eval] mAP50=%.4f mAP50-95=%.4f" % (res["mAP50"], res["mAP50-95"]))
    else:
        imgs = sorted([p for p in cfg["images"].iterdir()
                       if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
        totals = {t: {"tp": 0, "fp": 0, "fn": 0,
                      "per_class": collections.defaultdict(lambda: [0, 0, 0])} for t in CONFS}
        # 一次性低阈值推理，再按各阈值过滤（与 V2 threshold_sweep 同思路）
        stream = model.predict(source=[str(p) for p in imgs], imgsz=640, conf=min(CONFS),
                               iou=NMS_IOU, max_det=MAX_DET, device=0, stream=True,
                               verbose=False, batch=a.batch)
        n_img = 0
        boxes_at_report = 0
        no_pred_images = 0
        for p, r in zip(imgs, stream):
            n_img += 1
            import cv2
            im = cv2.imread(str(p))
            h, w = im.shape[:2]
            gts = [(c, ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
                   for (c, cx, cy, bw, bh) in load_gt(p)]
            if r.boxes is None or len(r.boxes) == 0:
                dets = []
            else:
                xyxy = r.boxes.xyxy.cpu().numpy()
                conf = r.boxes.conf.cpu().numpy()
                cls = r.boxes.cls.cpu().numpy().astype(int)
                dets = [(float(conf[i]), int(cls[i]), tuple(float(v) for v in xyxy[i]))
                        for i in range(len(conf))]
            if a.report_conf is not None:
                nb = sum(1 for d in dets if d[0] >= a.report_conf)
                boxes_at_report += nb
                if nb == 0:
                    no_pred_images += 1
            for t in CONFS:
                dt = [d for d in dets if d[0] >= t]
                tp, fp, fn = greedy_match(dt, gts, MATCH_IOU)
                totals[t]["tp"] += tp
                totals[t]["fp"] += fp
                totals[t]["fn"] += fn
                for cls_id in range(3):
                    d_c = [d for d in dt if d[1] == cls_id]
                    g_c = [g for g in gts if g[0] == cls_id]
                    a_, b_, c_ = greedy_match(d_c, g_c, MATCH_IOU)
                    pcc = totals[t]["per_class"][cls_id]
                    pcc[0] += a_; pcc[1] += b_; pcc[2] += c_
        rows = []
        for t in CONFS:
            tp, fp, fn = totals[t]["tp"], totals[t]["fp"], totals[t]["fn"]
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            rows.append({"conf": t, "tp": tp, "fp": fp, "fn": fn,
                         "precision": round(prec, 10), "recall": round(rec, 10),
                         "f1": round(f1, 10),
                         "per_class": {NAMES[k]: dict(zip(("tp", "fp", "fn"), v))
                                       for k, v in sorted(totals[t]["per_class"].items())}})
        res["sweep"] = rows
        if a.report_conf is not None:
            res["workpoint_stats"] = {
                "conf": a.report_conf, "n_images": n_img,
                "mean_boxes_per_image": round(boxes_at_report / n_img, 6) if n_img else None,
                "no_prediction_image_rate": round(no_pred_images / n_img, 6) if n_img else None,
                "no_prediction_images": no_pred_images}
            print("[eval] conf %.3f: mean boxes/image=%.3f, no-prediction rate=%.4f"
                  % (a.report_conf, boxes_at_report / n_img, no_pred_images / n_img))
        best = max(rows, key=lambda r: r["f1"])
        res["best_by_f1"] = best["conf"]
        print("[eval] 扫描完成；F1 最大的 conf = %.3f (F1 %.4f)" % (best["conf"], best["f1"]))
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("[eval] 写入 %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
