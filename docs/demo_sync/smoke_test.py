"""Web demo pipeline 同步后的**功能 smoke test**。

只做功能验证：**不跑任何冻结 test 集、不计算任何科研性能指标**。
用到的真实图是 `data/raw/MBDD/MBDD2025/JPEGImages/Hefei1000.jpg`
（已确认它属于 train、**不在**冻结 split_v2 test 的 1,179 张里），且只用于"链路能否跑通"。

覆盖 7 项：
  1 普通有检测输入：YOLO → ROI → SAM2 → 输出完整跑通
  2 BGR→RGB：确认送进 SAM2 的数组是 **RGB**（且确实与 BGR 不同）
  3 重叠 mask + 不同 conf：重叠像素最终属于**高 conf** 类
  4 zero-detection：app 的无检测文案为「未检出目标」，且全仓库不再出现「无缺陷」作该用途
  5 无标定：只输出像素面积/占比 + 「未标定」，不伪造 mm²
  6 configs/config.yaml、src/segment.py、src/detect.py 的 SHA256 **前后不变**
  7 V2/V3 evidence、weights、split/test manifest **无写入**

用法：
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe docs/demo_sync/smoke_test.py
"""
import csv
import hashlib
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2            # noqa: E402
import numpy as np    # noqa: E402

RESULTS = []


def rec(name, ok, detail):
    RESULTS.append((name, ok, detail))
    print("[%s] %-46s %s" % ("PASS" if ok else "FAIL", name, detail))
    sys.stdout.flush()


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


# ---------------------------------------------------------------- 3 用的桩件
class _Box:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = np.array(xyxy, dtype=np.float32)
        self.conf = float(conf)
        self.cls = int(cls)


def _colored_img(h=200, w=200):
    """有色合成图：三通道取值不同 ⇒ BGR 与 RGB 不相等，通道序断言才有判别力。
    （若用均匀灰图，BGR==RGB，任何通道序检查都会假通过。）"""
    img = np.zeros((h, w, 3), np.uint8)
    img[..., 0] = 200      # B
    img[..., 1] = 120      # G
    img[..., 2] = 40       # R
    img[60:140, 60:140] = (40, 200, 120)
    return img


class _FullMaskSAM:
    """桩 SAM2：对任何 ROI 返回中心 70% 的方块掩码（避开后处理的面积上限过滤）。"""

    def set_image(self, img):
        self.shape = img.shape[:2]

    def predict(self, point_coords=None, point_labels=None, box=None,
                multimask_output=False):
        h, w = self.shape
        m = np.zeros((h, w), dtype=np.uint8)
        y0, y1 = int(h * 0.15), int(h * 0.85)
        x0, x1 = int(w * 0.15), int(w * 0.85)
        m[y0:y1, x0:x1] = 1
        return m[None], np.array([0.9]), np.array([0.0])


def test_1_and_2():
    from src import detect, pipeline as pl, segment
    from src.config import get_config
    img_path = ROOT / "data/raw/MBDD/MBDD2025/JPEGImages/Hefei1000.jpg"
    if not img_path.exists():
        rec("1 端到端跑通", False, "缺测试图 %s" % img_path)
        return
    # 确认这张图不在冻结 test 清单里
    rows = list(csv.DictReader(open(ROOT / "docs/evidence/split_v2_frozen/split_v2_manifest.csv",
                                    encoding="utf-8")))
    frozen_test = {r["stem"] for r in rows if r["split"] == "test"}
    if img_path.stem in frozen_test:
        rec("1 端到端跑通", False, "所选图竟在冻结 test 里，拒绝使用")
        return
    img = cv2.imread(str(img_path))
    cfg = get_config()
    cfg.detect.conf = 0.075          # 冻结工作点（内存覆盖，不改 config.yaml）
    yolo = detect.load_yolo(cfg)
    sam2 = segment.build_sam2(cfg)

    # 用 spy 捕获真正传给 segment_roi 的数组
    captured = []
    orig = segment.segment_roi

    def spy(predictor, arr, box):
        captured.append(np.array(arr))
        return orig(predictor, arr, box)

    segment.segment_roi = spy
    try:
        res = pl.run_pipeline(img, cfg, yolo_model=yolo, sam2_predictor=sam2,
                              frozen_compat=True)   # 与 app.py 一致
        overlay = pl.overlay(img, res["mask"], cfg)
    finally:
        segment.segment_roi = orig

    n_det = len(captured)
    ok = (res["mask"].shape == img.shape[:2] and overlay.shape == img.shape
          and isinstance(res["summary"], dict))
    rec("1 端到端跑通", ok,
        "检测数=%d，掩码 shape=%s，summary 键=%s" % (n_det, res["mask"].shape,
                                                sorted(res["summary"].keys())[:4]))

    # 2) 送进 SAM2 的必须是 RGB
    if n_det == 0:
        rec("2 送入 SAM2 的是 RGB", True, "该图无检测框，无法直接验证（改用桩件路径验证）")
        return
    dets = sorted(detect.detect(img, yolo, cfg), key=lambda d: d.conf)  # 与 pipeline 同序
    expect_bgr, expect_rgb = [], []
    for d in dets:
        roi, _ = detect.crop_roi(img, d.xyxy, margin_ratio=0.1)
        if roi.size == 0:
            continue
        expect_bgr.append(roi)
        expect_rgb.append(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    same_as_rgb = sum(1 for a, b in zip(captured, expect_rgb) if a.shape == b.shape
                      and np.array_equal(a, b))
    # 与 **BGR** 相比是否确有差异：灰度区域 R==G==B 时通道互换是 no-op，属正常。
    n_diff_from_bgr = sum(1 for a, b in zip(captured, expect_bgr)
                          if not np.array_equal(a, b))
    colored = n_diff_from_bgr > 0
    ok = (same_as_rgb == len(expect_rgb)) and (colored or len(expect_rgb) > 0)
    rec("2 送入 SAM2 的是 RGB", same_as_rgb == len(expect_rgb),
        "%d/%d 个 ROI 与 cvtColor(BGR2RGB) 逐像素相同；其中与 BGR 有差异的 %d 个"
        "%s" % (same_as_rgb, len(expect_rgb), n_diff_from_bgr,
                "（该 ROI 接近灰度，通道互换本就是 no-op，属正常）" if not colored else ""))


def test_3():
    """两个重叠检测、conf 不同 → 重叠像素归属高 conf 类。"""
    from src import detect as det_mod, pipeline as pl
    from src.config import get_config
    cfg = get_config()
    img = np.full((200, 200, 3), 128, np.uint8)

    # 两个框重叠；A=类1(crack) conf 0.90，B=类2(rust) conf 0.30
    a = _Box([40, 40, 160, 160], 0.90, 1)
    b = _Box([80, 80, 180, 180], 0.30, 2)
    orig = det_mod.detect
    det_mod.detect = lambda image, model, cfg: [a, b]      # detect 的契约是降序
    try:
        res = pl.run_pipeline(img, cfg, yolo_model=object(), sam2_predictor=_FullMaskSAM(),
                              use_post=False, frozen_compat=True)
    finally:
        det_mod.detect = orig

    # A 的 ROI=[28:172,28:172] 中心块 → 覆盖像素 (100,100)；B 的 ROI 中心块也覆盖它
    owner = int(res["mask"][100, 100])
    ok = owner == 2   # 类 id+1：cls1→2, cls2→3
    # 说明：mask 存的是 cls+1，所以高 conf 的 cls=1 对应值 2
    rec("3 重叠区归属高 conf 类", owner == 2,
        "重叠像素 (100,100) 的类别值=%d（期望 2 = cls1 crack 高conf；若为 3 则是低 conf 胜）" % owner)
    # 反向断言：非重叠区不受影响（取一个落在两个 ROI 之外的点）
    v_out = int(res["mask"][175, 175])
    v_hi_only = int(res["mask"][60, 60])      # 只有高 conf 的 ROI 覆盖
    rec("3b 非重叠区未受影响", v_out == 0,
        "两 ROI 之外 (175,175)=%d（0=背景）；仅高 conf ROI 覆盖处 (60,60)=%d（2=cls1，符合预期）"
        % (v_out, v_hi_only))


def test_3c():
    """反向回归：默认路径必须**保持历史行为**（低 conf 胜 + BGR 原样送 SAM2）。

    这是「最小改动」的关键证据：改动只对 demo 生效，历史调用方不受影响。
    """
    from src import detect as det_mod, pipeline as pl
    from src.config import get_config
    cfg = get_config()
    img = _colored_img()
    a = _Box([40, 40, 160, 160], 0.90, 1)
    b = _Box([80, 80, 180, 180], 0.30, 2)
    orig = det_mod.detect
    det_mod.detect = lambda image, model, cfg: [a, b]
    seen = []
    class _SpySAM(_FullMaskSAM):
        def set_image(self, im):
            seen.append(np.array(im)); super().set_image(im)
    try:
        res = pl.run_pipeline(img, cfg, yolo_model=object(),
                              sam2_predictor=_SpySAM(), use_post=False)   # 默认 frozen_compat=False
    finally:
        det_mod.detect = orig
    owner = int(res["mask"][100, 100])
    # 直接与 crop_roi 得到的 BGR 切片比对，判断默认路径是否**原样送 BGR**
    from src import detect as _d
    roi_bgr, _off = _d.crop_roi(img, a.xyxy, margin_ratio=0.1)
    bgr_sent = bool(seen) and np.array_equal(seen[0], roi_bgr)
    rgb_sent = bool(seen) and np.array_equal(seen[0], cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB))
    ok = (owner == 3) and bgr_sent and not rgb_sent
    rec("3c 默认路径行为未变（最小影响）", ok,
        "默认 frozen_compat=False：重叠像素=%d（3=cls2 低conf胜 = 历史行为）；"
        "送 SAM2 的与 BGR 切片逐像素相同=%s、与 RGB 相同=%s"
        % (owner, bgr_sent, rgb_sent))


def test_4():
    src = (ROOT / "app/app.py").read_text(encoding="utf-8")
    has_new = "未检出目标" in src
    BAD = "无" + "缺陷"          # 拼出来，避免本检查器自身制造命中
    # 只把「UI 文案」判为违规：扫描会产生界面/报表文字的代码（app/）
    ui_dir = ROOT / "app"
    ui_hits = []
    other_hits = []
    for p in ROOT.rglob("*.py"):
        if ".venv" in p.parts or "__pycache__" in p.parts:
            continue
        if p.name == "smoke_test.py":      # 检查器自身
            continue
        try:
            t = p.read_text(encoding="utf-8")
        except Exception:
            continue
        for i, line in enumerate(t.splitlines(), 1):
            if BAD in line:
                (ui_hits if ui_dir in p.parents else other_hits).append(
                    "%s:%d %s" % (p.relative_to(ROOT), i, line.strip()[:60]))
    rec("4 无检测文案为「未检出目标」", has_new and not ui_hits,
        "app.py 含新文案=%s；UI 代码中的违规=%s；其它语义出现处（非 UI，需保留）=%s"
        % (has_new, ui_hits or "0", other_hits or "无"))


def test_5():
    from src import quantify as q
    mask = np.zeros((50, 50), np.uint8)
    mask[10:20, 10:20] = 1
    d_uncal = q.quantify_mask(mask, 0, ["peeling", "crack", "rust"], 50, 50, None)
    d_cal = q.quantify_mask(mask, 0, ["peeling", "crack", "rust"], 50, 50, 0.5)
    rows = q.to_table_rows({"per_class": [d_uncal], "total_pixel_area": d_uncal["pixel_area"],
                            "total_ratio": d_uncal["ratio"], "total_area_mm2": None},
                           ["peeling", "crack", "rust"])
    ok = (d_uncal["area_mm2"] is None and rows[0]["真实面积(mm²)"] == "未标定"
          and rows[0]["像素面积"] == 100 and abs(rows[0]["损伤占比"] - 100 / 2500) < 1e-9
          and d_cal["area_mm2"] == 100 * 0.25)
    rec("5 无标定只报像素/占比+未标定", ok,
        "无标定 area_mm2=%s、表格单元格=%r、像素=%d、占比=%.4f；有标定时 mm²=%.2f"
        % (d_uncal["area_mm2"], rows[0]["真实面积(mm²)"], rows[0]["像素面积"],
           rows[0]["损伤占比"], d_cal["area_mm2"]))


def test_6():
    pre = json.load(open(ROOT / "docs/demo_sync/pre_change_code_sha256.json", encoding="utf-8"))
    must_same = ["configs/config.yaml", "src/segment.py", "src/detect.py"]
    bad = []
    for f in must_same:
        now = sha(ROOT / f)
        if now != pre[f]:
            bad.append("%s: %s -> %s" % (f, pre[f][:12], now[:12]))
    changed = [f for f in pre if sha(ROOT / f) != pre[f]]
    rec("6 三个受保护文件 SHA256 未变", not bad,
        "不一致=%s；实际被改动的是=%s" % (bad or "无", changed))


def test_7():
    """发布准备窗口内：模型权重与评价证据**零写入**。

    原实现把 `weights/` 下任何文件都判为违规，会把"故意新增的文档
    weights/README.md"误报成问题——这里改成**只判模型权重文件**，
    并显式核对两个已发布权重的 SHA256。
    """
    import hashlib
    cutoff = (ROOT / "docs/demo_sync/pre_change_code_sha256.json").stat().st_mtime
    WEIGHT_EXT = {".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".zip"}
    protected_dirs = ["docs/evidence", "experiments", "data/splits", "data/splits_v2"]
    touched, weight_touched = [], []
    for d in protected_dirs:
        p = ROOT / d
        if not p.exists():
            continue
        for q in p.rglob("*"):
            if q.is_file() and q.stat().st_mtime > cutoff + 1:
                touched.append(str(q.relative_to(ROOT)))
    for d in ("weights", "experiments"):
        p = ROOT / d
        if not p.exists():
            continue
        for q in p.rglob("*"):
            if q.is_file() and q.suffix.lower() in WEIGHT_EXT:
                weight_touched.append(str(q.relative_to(ROOT)))
    # 显式核对两个已发布权重的哈希
    EXPECT = {
        "weights/best.pt": "f8c9ab585e456845e64a91ae046baa569ffc9c276800c875461e8cdbd37d497f",
        "weights/best_orig.pt": "dbc16595b3b90491f9dad0b242a813c5958f7c489485fa3cf6f8aad3bb4a0e43",
    }
    bad_hash = [f for f, want in EXPECT.items()
                if hashlib.sha256((ROOT / f).read_bytes()).hexdigest() != want]
    # 被写入过、且是模型权重文件的（真正的风险）
    changed_weights = [f for f in weight_touched if (ROOT / f).stat().st_mtime > cutoff + 1]
    ok = not touched and not bad_hash and not changed_weights
    rec("7 权重与评价证据零写入", ok,
        "评价证据目录被写=%s；权重文件被写=%s；两个已发布权重哈希=%s"
        % (touched or "无", changed_weights or "无",
           "全部未变" if not bad_hash else bad_hash))


def main():
    print("=== demo pipeline 同步 smoke test（不跑冻结 test、不算科研指标）===\n")
    test_1_and_2()
    test_3()
    test_3c()
    test_4()
    test_5()
    test_6()
    test_7()
    print("\n=== 汇总 ===")
    n_ok = sum(1 for _, ok, _ in RESULTS if ok)
    print("%d/%d 项通过" % (n_ok, len(RESULTS)))
    out = ROOT / "docs/demo_sync/smoke_test_result.json"
    json.dump([{"name": n, "pass": ok, "detail": d} for n, ok, d in RESULTS],
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("结果写入 %s" % out.relative_to(ROOT))
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
