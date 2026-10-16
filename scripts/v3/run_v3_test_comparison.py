"""Smart Campus V3 test 的**一次性**比较运行（Baseline V2 vs Adapted V3）。

⚠️ 本脚本会读取 test。使用前提（脚本自己会强制检查）：
  1. `--unsealed-by <真人姓名>` 必须给出（agent 不代填 unsealed_by）；
  2. `v3_test_comparison_protocol.md` 的 SHA256 必须与 sidecar 一致（协议未被改动）；
  3. 先写 `v3_test_unseal.json`（存在则只读不覆盖），**然后**才构造模型 ——
     保证"拆封记录写在第一次 test 推理之前"。

一次运行、两个模型，评价参数逐项相同：
  conf 0.001 / NMS iou 0.7 / max_det 300 / imgsz 640 / 同一 evaluator / 同一 test manifest

用法（由用户授权后执行）：
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe \
    docs/evidence/v3_domain_adaptation/_scripts/run_v3_test_comparison.py \
    --unsealed-by <姓名> --write
"""
import argparse
import csv
import hashlib
import json
import pathlib
import subprocess
import sys
import time

NL = chr(10)
ROOT = pathlib.Path(__file__).resolve().parents[4]
EV = ROOT / "docs/evidence/v3_domain_adaptation"
MF = EV / "model_frozen"
EVAL = EV / "_scripts/eval_v3_checkpoints.py"
PROTO = EV / "v3_test_comparison_protocol.md"
PROTO_SIDE = EV / "v3_test_comparison_protocol.sha256"
UNSEAL = MF / "v3_test_unseal.json"
TEST_MANIFEST = EV / "smart_campus/v3_split/smart_campus_v3_test_manifest.csv"
V2W = ROOT / "experiments/runs_v2/V2-AUG-RERUN1/weights/best.pt"
V3W = ROOT / "experiments/runs_v3/V3-CAMPUS-ADAPT/weights/epoch26.pt"   # selected epoch 27
V3_CONF = 0.20


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def sidecar_hash(p):
    return p.read_text(encoding="utf-8").strip().split("=")[-1].strip()


def run(cmd, label):
    print("[run] %s" % label)
    sys.stdout.flush()
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(r.stdout[-800:])
    if r.returncode != 0:
        print("[FAIL] %s%s%s" % (label, NL, r.stderr[-1500:]))
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unsealed-by", required=True)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    # ---- 前置检查（任一不过则**不写、不跑**）----
    if sha(PROTO) != sidecar_hash(PROTO_SIDE):
        print("[FATAL] 比较协议与 sidecar 不一致，协议被改动过——拒绝运行")
        return 2
    v2_thr = json.load(open(MF / "v2_on_sc_val_baseline.json", encoding="utf-8"))["frozen_threshold"]
    rec = {
        "unseal_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "unsealed_by": a.unsealed_by,
        "smart_campus_v3_test_manifest_sha256": sha(TEST_MANIFEST),
        "v2_weights_path": str(V2W), "v2_weights_sha256": sha(V2W),
        "v3_weights_path": str(V3W), "v3_weights_sha256": sha(V3W),
        "v3_selected_epoch": 27,
        "v2_frozen_threshold": v2_thr,
        "v3_frozen_threshold": V3_CONF,
        "evaluator_sha256": sha(EVAL),
        "comparison_protocol_sha256": sha(PROTO),
        "note": "本记录写在第一次 test 推理之前；test 只用于一次性报告，不参与任何选择",
    }
    if not a.write:
        print(json.dumps(rec, ensure_ascii=False, indent=2))
        print(NL + "[dry-run] 未写任何文件、未读取任何 test")
        return 0
    MF.mkdir(parents=True, exist_ok=True)
    if UNSEAL.exists():
        print("[info] 拆封记录已存在，只读不覆盖：%s" % UNSEAL)
    else:
        json.dump(rec, open(UNSEAL, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("[UNSEAL] 已写拆封记录：%s（unsealed_by=%s）" % (UNSEAL, a.unsealed_by))

    out = EV / "v3_test_eval"
    out.mkdir(parents=True, exist_ok=True)
    res = {"unseal": rec, "models": {}}
    for tag, ck, conf in (("v2_baseline", V2W, v2_thr), ("v3_adapted", V3W, V3_CONF)):
        run([sys.executable, str(EVAL), "--ckpt", str(ck), "--set", "sc_test", "--mode", "map",
             "--out", str(out / ("%s_map.json" % tag))], "%s mAP" % tag)
        run([sys.executable, str(EVAL), "--ckpt", str(ck), "--set", "sc_test", "--mode", "sweep",
             "--report-conf", str(conf), "--out", str(out / ("%s_sweep.json" % tag))],
            "%s 工作点 + 扫描" % tag)
        m = json.load(open(out / ("%s_map.json" % tag), encoding="utf-8"))
        s = json.load(open(out / ("%s_sweep.json" % tag), encoding="utf-8"))
        wp = [r for r in s["sweep"] if abs(r["conf"] - conf) < 1e-9][0]
        res["models"][tag] = {"weights": str(ck), "weights_sha256": sha(ck),
                              "frozen_conf": conf,
                              "map": {"mAP50": m["mAP50"], "mAP50-95": m["mAP50-95"],
                                      "per_class": m["per_class"]},
                              "workpoint": {"P": wp["precision"], "R": wp["recall"], "F1": wp["f1"],
                                            "per_class": wp["per_class"],
                                            **s.get("workpoint_stats", {})}}
    json.dump(res, open(out / "v3_test_comparison_results.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    with open(out / "v3_test_reruns.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rerun_timestamp", "script", "reason", "operator", "prior_result_kept", "command"])
    print("[DONE] 结果 → %s/v3_test_comparison_results.json" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
