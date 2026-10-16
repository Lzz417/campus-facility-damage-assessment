"""冻结 V3-CAMPUS-ADAPT 的最终模型（**在第一次 V3 test 推理之前**）。

产物目录：`docs/evidence/v3_domain_adaptation/model_frozen/`
  v3_model_freeze.json                 结构化总记录
  v3_model_freeze_manifest.csv          逐件 SHA256（含模型权重副本？不——只记哈希，不复制大文件）
  v3_model_freeze.sha256                上面两个的自校验
  v3_model_frozen_summary.md            人读摘要
  selected_checkpoint_scores.csv        逐 epoch 两域对照（来自 per_epoch_eval）

必须写明：
  Smart Campus V3 test status: NOT USED
  split_v2 test status in V3: NOT USED
  frozen_before_first_v3_test_inference: true

用法：
  ... freeze_v3_model.py --epoch 12 --conf 0.075 --threshold-sweep <sweep.json> --write
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
ROOT = pathlib.Path(__file__).resolve().parents[4]
RUN = ROOT / "experiments/runs_v3/V3-CAMPUS-ADAPT"
EV = ROOT / "docs/evidence/v3_domain_adaptation"
OUT = EV / "model_frozen"
INIT_W = ROOT / "experiments/runs_v2/V2-AUG-RERUN1/weights/best.pt"


def sha(p, chunk=1 << 20):
    p = pathlib.Path(p)
    if not p.exists():
        return "MISSING"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def line_of(path):
    t = pathlib.Path(path).read_text(encoding="utf-8").strip()
    return t.split("=")[-1].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=int, required=True)
    ap.add_argument("--conf", type=float, required=True, help="在 Smart Campus V3 val 上定下的工作点阈值")
    ap.add_argument("--threshold-sweep", required=True)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    ck = RUN / "weights" / ("epoch%d.pt" % (a.epoch - 1))   # epochN.pt 的 N 从 0 起
    if not ck.exists():
        ck = RUN / "weights" / ("epoch%d.pt" % a.epoch)
    if not ck.exists():
        print("[FATAL] 找不到 epoch %d 的权重" % a.epoch)
        return 2

    PE = EV / "v3_training/per_epoch_eval"
    A = json.load(open(PE / ("epoch%d_mbdd_val_map.json" % a.epoch), encoding="utf-8"))
    B = json.load(open(PE / ("epoch%d_sc_val_map.json" % a.epoch), encoding="utf-8"))
    sweep = json.load(open(a.threshold_sweep, encoding="utf-8"))
    sel = json.load(open(EV / "v3_training/selection.json", encoding="utf-8"))

    train_args = RUN / "args.yaml"
    scripts = sorted((EV / "_scripts").glob("*.py"))
    rec = {
        "freeze_type": "v3_campus_adapt_model_freeze",
        "freeze_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "run_name": "V3-CAMPUS-ADAPT",
        "framing": "continued domain-adaptation / fine-tuning from V2 (NOT training from official pretrained weights)",
        "selected_epoch": a.epoch,
        "selected_weights_path": str(ck),
        "selected_weights_sha256": sha(ck),
        "init_weights_path": str(INIT_W),
        "init_weights_sha256": sha(INIT_W),
        "train_data": {"mbdd_split_v2_train_orig": 7752, "smart_campus_v3_train": 1536, "total": 9288},
        "combined_train_manifest_sha256": line_of(EV / "v3_training/combined_train_manifest.sha256"),
        "v3_split_freeze_manifest_sha256": line_of(EV / "smart_campus/v3_split/v3_split_freeze.sha256")
                                            if (EV / "smart_campus/v3_split/v3_split_freeze.sha256").exists() else "MISSING",
        "v3_split_manifest_csv_sha256": sha(EV / "smart_campus/v3_split/smart_campus_v3_split_manifest.csv"),
        "model_selection_protocol_sha256": line_of(EV / "v3_model_selection_protocol.sha256"),
        "train_args_path": str(train_args),
        "train_args_sha256": sha(train_args),
        "scripts_sha256": {p.name: sha(p) for p in scripts},
        "dev_set_metrics": {
            "A_mbdd_split_v2_val": {"n": 2833, "mAP50": A["mAP50"], "mAP50-95": A["mAP50-95"],
                                    "P": A["P"], "R": A["R"], "per_class": A["per_class"]},
            "B_smart_campus_v3_val": {"n": 329, "mAP50": B["mAP50"], "mAP50-95": B["mAP50-95"],
                                      "P": B["P"], "R": B["R"], "per_class": B["per_class"]},
        },
        "selection": {"rule": "protocol v3_model_selection_protocol.md",
                      "guard_mbdd_val_map50_min": sel["protocol_guard"],
                      "n_passed_guard": sel["n_passed_guard"],
                      "selected": {k: sel["selected"][k] for k in
                                   ("epoch", "A_mAP50", "A_mAP50_95", "B_mAP50", "B_mAP50_95")}},
        "operating_point": {"frozen_conf_threshold": a.conf,
                            "calibrated_on": "Smart Campus V3 val (329)",
                            "rule": "F1 max; tie within 0.005 -> higher recall -> higher threshold",
                            "sweep": sweep.get("sweep")},
        "eval_params": {"imgsz": 640, "nms_iou": 0.45, "max_det": 30, "match_iou": 0.5,
                        "map_protocol": {"conf": 0.001, "iou": 0.7, "max_det": 300}},
        "class_mapping": {"paint_degradation_to": 0, "cracks_to": 1, "corrosion_to": 2,
                          "names": {"0": "peeling", "1": "crack", "2": "rust"}},
        "discipline": {
            "smart_campus_v3_test_status": "NOT USED",
            "split_v2_test_status_in_v3": "NOT USED",
            "frozen_before_first_v3_test_inference": True,
        },
    }

    if not a.write:
        print(json.dumps(rec, ensure_ascii=False, indent=2)[:1200])
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    # 逐 epoch 对照表
    src = EV / "v3_training/selection_table.csv"
    if src.exists():
        shutil.copy2(src, OUT / "selected_checkpoint_scores.csv")
    json.dump(rec, open(OUT / "v3_model_freeze.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    arts = []
    for p in sorted(OUT.iterdir()):
        if p.is_file() and p.name not in ("v3_model_freeze.sha256", "v3_model_freeze_manifest.csv"):
            arts.append({"path": p.name, "sha256": sha(p), "size_bytes": p.stat().st_size})
    with open(OUT / "v3_model_freeze_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "sha256", "size_bytes"])
        w.writeheader()
        w.writerows(arts)
    with open(OUT / "v3_model_freeze.sha256", "w", encoding="utf-8") as f:
        for n in ("v3_model_freeze.json", "v3_model_freeze_manifest.csv"):
            f.write("%s  SHA256 = %s%s" % (n, sha(OUT / n), NL))
    print("[FREEZE] epoch %d、conf %.3f；%d 件 → %s" % (a.epoch, a.conf, len(arts), OUT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
