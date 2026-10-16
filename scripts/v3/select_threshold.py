"""工作点阈值选择规则（**V2 baseline 与 V3 共用同一段代码**）。

规则（与 `v3_model_selection_protocol.md` 第 4 节逐字对应）：
  primary  = F1 最大
  tie band = 与最大 F1 相差 <= 0.005 的阈值都算并列
  tie break= 并列中取 **recall 更高**者
  final tie= 仍相同则取 **较高阈值**

抽成独立函数的目的：V3 的 conf=0.20 当初是我手算的；
这里用同一个函数把 V3 的结果**重新推导一遍**，若复现不出 0.20，
说明"同一套规则"是嘴上说说。V2 的阈值也用这个函数定。

用法（作为库）：
    from select_threshold import select_threshold
    thr, detail = select_threshold(sweep_rows)
"""
TIE_BAND = 0.005


def select_threshold(rows):
    """rows: [{'conf': float, 'precision': float, 'recall': float, 'f1': float}, ...]"""
    if not rows:
        raise ValueError("空的扫描结果")
    best_f1 = max(r["f1"] for r in rows)
    tied = [r for r in rows if best_f1 - r["f1"] <= TIE_BAND]
    hi_rec = max(r["recall"] for r in tied)
    tied2 = [r for r in tied if hi_rec - r["recall"] <= 1e-12]
    chosen = max(tied2, key=lambda r: r["conf"])
    return chosen["conf"], {
        "rule": "primary=F1 max; tie band<=0.005; then higher recall; final tie higher conf",
        "tie_band": TIE_BAND,
        "f1_max": best_f1,
        "f1_argmax_conf": max(rows, key=lambda r: r["f1"])["conf"],
        "n_tied_on_primary": len(tied),
        "tied_confs_on_primary": [r["conf"] for r in tied],
        "n_tied_on_recall": len(tied2),
        "tied_confs_on_recall": [r["conf"] for r in tied2],
        "selected_conf": chosen["conf"],
        "selected_row": chosen,
    }


if __name__ == "__main__":
    import json
    import sys
    d = json.load(open(sys.argv[1], encoding="utf-8"))
    thr, det = select_threshold(d["sweep"])
    print(json.dumps(det, ensure_ascii=False, indent=2))
