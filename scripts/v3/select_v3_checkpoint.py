"""按**预登记**规则从 V3 的逐 epoch checkpoint 中选出最终模型。

规则来源：`docs/evidence/v3_domain_adaptation/v3_model_selection_protocol.md`
（该文件在训练启动前写定并算了 SHA256；本脚本只执行，不重新定义规则。）

  1. forgetting guard：`A_mAP50 >= 0.0948`（= V2 基线 0.1185 x 0.80）才算合格候选；
  2. primary：合格候选中 `B_mAP50` 最大者；
  3. tie-break：与最大值相差 <= 0.005 的视为并列 →
     (1) 取 `B_mAP50-95` 更高者；(2) 仍相差 <= 0.005 则取**更早的 epoch**；
  4. 若无任何 checkpoint 满足 guard → 打印
     `no checkpoint passed the pre-registered forgetting guard` 并以非零码退出。

A = MBDD split_v2 val (2833)，B = Smart Campus V3 val (329)。

用法：
  ... select_v3_checkpoint.py --eval-dir <per-epoch jsons 目录> --out selection.json
"""
import argparse
import csv
import json
import pathlib
import re
import sys

GUARD = 0.0948            # 0.1185 x 0.80，见协议
TIE = 0.005
BASELINE = 0.1185


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True, help="含 *_mbdd_val_map.json / *_sc_val_map.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--md-out", default=None)
    a = ap.parse_args()

    E = pathlib.Path(a.eval_dir)
    rows = []
    # 硬约束：只接受 epoch<数字> 的候选文件。best.pt / last.pt 的一致性核对结果
    # 走独立目录与独立命名，**不可能**被这条 glob 收进来（防止变成第 31/32 个候选）。
    CANDS = sorted(f for f in E.glob("epoch*_mbdd_val_map.json")
                   if re.fullmatch(r"epoch\d+_mbdd_val_map\.json", f.name))
    skipped = [f.name for f in E.glob("*_mbdd_val_map.json") if f not in CANDS]
    if skipped:
        print("[info] 忽略非 epoch 候选文件：%s" % skipped)
    for f in CANDS:
        stem = f.name[: -len("_mbdd_val_map.json")]
        g = E / (stem + "_sc_val_map.json")
        if not g.exists():
            print("[warn] 缺 %s 的 sc_val 结果，跳过" % stem)
            continue
        A = json.load(open(f, encoding="utf-8"))
        B = json.load(open(g, encoding="utf-8"))
        ep = int(stem.split("epoch")[-1].replace("_", "")) if "epoch" in stem else None
        rows.append({"checkpoint": stem, "epoch": ep,
                     "A_mAP50": A["mAP50"], "A_mAP50_95": A["mAP50-95"],
                     "B_mAP50": B["mAP50"], "B_mAP50_95": B["mAP50-95"],
                     "A_per_class": A.get("per_class"), "B_per_class": B.get("per_class"),
                     "ckpt_path": A["checkpoint"], "ckpt_sha256": A["checkpoint_sha256"]})
    if not rows:
        print("[FATAL] 没找到任何成对的评价结果")
        return 2
    rows.sort(key=lambda r: (r["epoch"] is None, r["epoch"]))

    passed = [r for r in rows if r["A_mAP50"] >= GUARD]
    out = {"protocol_guard": GUARD, "protocol_baseline_mbdd_val_map50": BASELINE,
           "n_checkpoints": len(rows), "n_passed_guard": len(passed),
           "rows": rows}
    if not passed:
        out["decision"] = "no checkpoint passed the pre-registered forgetting guard"
        json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("no checkpoint passed the pre-registered forgetting guard")
        print("（最佳 A_mAP50 = %.4f，仍低于阈值 %.4f）"
              % (max(r["A_mAP50"] for r in rows), GUARD))
        print("按协议：不放宽规则、不解封 test，停止并交用户决定。")
        return 3

    best = max(r["B_mAP50"] for r in passed)
    tied = [r for r in passed if best - r["B_mAP50"] <= TIE]
    best95 = max(r["B_mAP50_95"] for r in tied)
    tied2 = [r for r in tied if best95 - r["B_mAP50_95"] <= TIE]
    chosen = min(tied2, key=lambda r: r["epoch"])
    out.update({"decision": "selected", "B_mAP50_max": best,
                "n_tied_on_primary": len(tied), "n_tied_on_secondary": len(tied2),
                "selected": chosen})
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    L = ["# V3-CAMPUS-ADAPT 逐 epoch 两域对照与选择结果", "",
         "guard：MBDD val mAP50 ≥ **%.4f**（= V2 基线 %.4f × 0.80）；primary：Smart Campus val mAP50 最大" % (GUARD, BASELINE), "",
         "| epoch | MBDD val mAP50 | MBDD val mAP50-95 | SC val mAP50 | SC val mAP50-95 | 过 guard |",
         "|---|---|---|---|---|---|"]
    for r in rows:
        L.append("| %s | %.4f | %.4f | %.4f | %.4f | %s |"
                 % (r["epoch"], r["A_mAP50"], r["A_mAP50_95"], r["B_mAP50"], r["B_mAP50_95"],
                    "✅" if r["A_mAP50"] >= GUARD else "✗"))
    L += ["", "**选中**：epoch %s（SC val mAP50 %.4f / mAP50-95 %.4f；MBDD val mAP50 %.4f）"
          % (chosen["epoch"], chosen["B_mAP50"], chosen["B_mAP50_95"], chosen["A_mAP50"]),
          "", "并列情况：primary 并列 %d 个、secondary 并列 %d 个，按规则取最早 epoch。"
          % (len(tied), len(tied2))]
    if a.md_out:
        pathlib.Path(a.md_out).write_text("\n".join(L) + "\n", encoding="utf-8")
    with open(str(a.out).replace(".json", "_table.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "A_mbdd_val_map50", "A_mbdd_val_map50_95",
                    "B_sc_val_map50", "B_sc_val_map50_95", "passed_guard"])
        for r in rows:
            w.writerow([r["epoch"], r["A_mAP50"], r["A_mAP50_95"], r["B_mAP50"],
                        r["B_mAP50_95"], int(r["A_mAP50"] >= GUARD)])
    print("选中 epoch %s：SC val mAP50 %.4f、MBDD val mAP50 %.4f（过 guard）"
          % (chosen["epoch"], chosen["B_mAP50"], chosen["A_mAP50"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
