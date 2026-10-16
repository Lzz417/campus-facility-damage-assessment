"""冻结 **V2-AUG-RERUN1 在 Smart Campus V3 val 上的固定 baseline**。

不训练、不改权重、不碰 test —— 只是用与 V3 **完全相同**的 post-hoc evaluator
在**同一个 val 集**上把 V2 测一遍，并用**同一段**选阈函数定一个只用于本次比较的工作点。

产物（写到 model_frozen/）：
  v2_on_sc_val_baseline.json          结构化记录（含权重/evaluator/val manifest 的 SHA256）
  v2_on_sc_val_threshold_scan.csv     10 点扫描明细
  v2_on_sc_val_baseline_summary.md    人读摘要

用法：
  ... freeze_v2_sc_val_baseline.py --write
"""
import argparse
import csv
import hashlib
import json
import pathlib
import sys
import time

NL = chr(10)
ROOT = pathlib.Path(__file__).resolve().parents[4]
EV = ROOT / "docs/evidence/v3_domain_adaptation"
MF = EV / "model_frozen"
V2W = ROOT / "experiments/runs_v2/V2-AUG-RERUN1/weights/best.pt"
MAP_JSON = MF / "v2_on_sc_val_baseline.json"
SWEEP_JSON = EV / "v3_training/v2_on_sc_val_sweep.json"
VAL_LIST = EV / "smart_campus/v3_split/smart_campus_v3_val.txt"
FREEZE_MANIFEST = EV / "smart_campus/v3_split/v3_split_freeze_manifest.csv"

sys.path.insert(0, str(EV / "_scripts"))
from select_threshold import select_threshold  # noqa: E402


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def sha_of_in_manifest(name):
    for r in csv.DictReader(open(FREEZE_MANIFEST, encoding="utf-8")):
        if r["path"].rstrip("/") == name:
            return r["sha256"]
    return "NOT_FOUND"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    base = json.load(open(MAP_JSON, encoding="utf-8"))
    sweep = json.load(open(SWEEP_JSON, encoding="utf-8"))
    thr, detail = select_threshold(sweep["sweep"])
    print("[V2 baseline] SC val mAP50=%.4f mAP50-95=%.4f ；选定 conf=%.3f"
          % (base["mAP50"], base["mAP50-95"], thr))
    print("              依据：F1 max@%.3f；并列 %s → recall 最高者 → conf %.2f"
          % (detail["f1_argmax_conf"], detail["tied_confs_on_primary"], thr))

    rec = {
        "freeze_type": "v2_baseline_on_smart_campus_v3_val",
        "freeze_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "purpose": "为 Smart Campus V3 test 上的 V2 vs V3 比较，提供一个冻结的、只依据 val 的 baseline 工作点",
        "model": "V2-AUG-RERUN1 (V2 正式 detector)",
        "weights_path": str(V2W),
        "weights_sha256": sha(V2W),
        "training_this_round": False,
        "weights_modified": False,
        "evaluator_path": str(EV / "_scripts/eval_v3_checkpoints.py"),
        "evaluator_sha256": sha(EV / "_scripts/eval_v3_checkpoints.py"),
        "evaluator_used_for_v3_too": True,
        "val_list_path": str(VAL_LIST),
        "val_manifest_sha256": sha_of_in_manifest("smart_campus_v3_val.txt"),
        "val_list_file_sha256": sha(VAL_LIST),
        "val_n": 329,
        "standard_map_protocol": {"conf": 0.001, "iou": 0.7, "max_det": 300, "imgsz": 640},
        "metrics": {"mAP50": base["mAP50"], "mAP50-95": base["mAP50-95"],
                    "P": base["P"], "R": base["R"], "per_class": base["per_class"]},
        "threshold_selection": detail,
        "frozen_threshold": thr,
        "threshold_scan_path": str(MF / "v2_on_sc_val_threshold_scan.csv"),
        "tie_break_shared_with_v3": True,
        "rule_source": "v3_model_selection_protocol.md 第 4 节；由 _scripts/select_threshold.py 实现（V2/V3 共用）",
        "discipline": {"smart_campus_v3_test_status": "NOT USED",
                       "split_v2_test_status_in_v3": "NOT USED"},
    }

    if not a.write:
        print(json.dumps(rec, ensure_ascii=False, indent=2)[:900])
        return 0

    json.dump(rec, open(MF / "v2_on_sc_val_baseline.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    with open(MF / "v2_on_sc_val_threshold_scan.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["conf", "tp", "fp", "fn", "precision", "recall", "f1",
                    "selected", "peeling_tp", "peeling_fp", "peeling_fn",
                    "crack_tp", "crack_fp", "crack_fn", "rust_tp", "rust_fp", "rust_fn"])
        for r in sweep["sweep"]:
            pc = r.get("per_class", {})
            g = lambda c, k: pc.get(c, {}).get(k, "")  # noqa: E731
            w.writerow([r["conf"], r["tp"], r["fp"], r["fn"], r["precision"], r["recall"],
                        r["f1"], int(abs(r["conf"] - thr) < 1e-9),
                        g("peeling", "tp"), g("peeling", "fp"), g("peeling", "fn"),
                        g("crack", "tp"), g("crack", "fp"), g("crack", "fn"),
                        g("rust", "tp"), g("rust", "fp"), g("rust", "fn")])
    sel = [r for r in sweep["sweep"] if abs(r["conf"] - thr) < 1e-9][0]
    L = ["# V2 baseline on Smart Campus V3 val（冻结）", "",
         "**用途**：为 Smart Campus V3 test 上的 **V2 vs V3 比较**提供一个冻结的 baseline。",
         "**本轮未训练、未修改权重**；只在 **V3 val（329 张）** 上做评价与选阈，**未读任何 test**。", "",
         "| 项 | 值 |", "|---|---|",
         "| 模型 | V2-AUG-RERUN1（V2 正式 detector） |",
         "| 权重 | `%s` |" % V2W,
         "| 权重 SHA256 | `%s` |" % rec["weights_sha256"],
         "| evaluator SHA256 | `%s`（**与 V3 用的是同一个脚本**） |" % rec["evaluator_sha256"],
         "| val manifest SHA256 | `%s` |" % rec["val_manifest_sha256"],
         "| 冻结时间 | `%s` |" % rec["freeze_timestamp"], "",
         "## 标准 mAP（与 V3 完全同口径：conf 0.001 / iou 0.7 / max_det 300 / imgsz 640）", "",
         "| 指标 | V2 baseline | V3-CAMPUS-ADAPT (epoch 27) |", "|---|---|---|",
         "| **mAP50** | **%.4f** | 0.6995 |" % base["mAP50"],
         "| **mAP50-95** | **%.4f** | 0.4671 |" % base["mAP50-95"], "",
         "逐类 AP50：", "", "| 类 | V2 | V3 |", "|---|---|---|"]
    v3pc = {"peeling": 0.7778, "crack": 0.8571, "rust": 0.4636}
    for c in base["per_class"]:
        L.append("| %s | %.4f | %.4f |" % (c["class_name"], c["AP50"], v3pc.get(c["class_name"], float("nan"))))
    L += ["", "## 工作点阈值（与 V3 同一套规则、同一段代码）", "",
          "- primary = F1 最大：**%.3f**（conf %.3f）" % (detail["f1_max"], detail["f1_argmax_conf"]),
          "- 并列带 ≤0.005 的阈值：%s" % detail["tied_confs_on_primary"],
          "- tie-break → recall 更高者：%s" % detail["tied_confs_on_recall"],
          "- **冻结阈值 = conf %.3f**（V2）／ conf 0.200（V3，已冻结）" % thr, "",
          "该阈值下：P %.4f / R %.4f / F1 %.4f"
          % (sel["precision"], sel["recall"], sel["f1"]), "",
          "> 两个模型的阈值**都是在同一个 Smart Campus V3 val 上、按同一条规则分别校准**的，",
          "> 因此工作点指标可以并列比较，但必须带上这句说明。", "",
          "## 纪律", "",
          "```", "Smart Campus V3 test status: NOT USED", "split_v2 test status in V3: NOT USED", "```"]
    (MF / "v2_on_sc_val_baseline_summary.md").write_text(NL.join(L) + NL, encoding="utf-8")
    print("[写] 3 个文件 → %s" % MF)
    return 0


if __name__ == "__main__":
    sys.exit(main())
