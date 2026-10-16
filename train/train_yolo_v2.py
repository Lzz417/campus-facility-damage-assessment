# -*- coding: utf-8 -*-
"""split_v2 训练脚本（Request 9 / 第八、九节）。

在**冻结的 split_v2** 上重训 YOLO11n 检测器，两个 run：
  V2-ORIG —— data/splits_v2/data_orig.yaml（train = images/train_orig，未增强）
  V2-AUG  —— data/splits_v2/data_aug.yaml （train = images/train，原始 + 六类退化）

两者都从**同一个官方** weights/yolo11n.pt 初始化。

与历史实验保持一致的主配置：epochs=100, batch=32, imgsz=640, device=0,
deterministic=True, seed=0（与 runs/detect/train_aug 的历史 args.yaml 一致）。

纪律：
  * **不使用 test**：训练只做 val（val=True），绝不传入 test 做评价；
  * 不覆盖 weights/best.pt、weights/best_orig.pt、data/splits/ 与任何历史实验文件；
  * 结果写入 experiments/runs_v2/<name>/。

训练前（本节第八项）保存：config、Python 环境、GPU、code commit、
split_v2 manifest hash、data yaml hash、seed、模型初始化、起止时间。
"""
from __future__ import annotations

import argparse, hashlib, json, platform, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

# 仓库根 = 本文件的上一级目录（train/ 位于仓库根下）。
# 原实现此处硬编码了本机绝对路径，已改为按文件位置推断，便于他人复现。
ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "experiments" / "runs_v2"
FLZ = ROOT / "docs" / "evidence" / "split_v2_frozen"

MAIN_ARGS = dict(epochs=100, batch=32, imgsz=640, device="0",
                 workers=8, seed=0, deterministic=True, cache=False,
                 optimizer="auto", patience=100, amp=True, val=True,
                 close_mosaic=10, plots=True)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sh(*cmd) -> str:
    try:
        return subprocess.check_output(list(cmd), cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "UNAVAILABLE"


def env_snapshot() -> dict:
    import importlib.metadata as md
    info = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": {},
    }
    for p in ("torch", "torchvision", "ultralytics", "opencv-python",
              "numpy", "pillow"):
        try:
            info["packages"][p] = md.version(p)
        except Exception:
            info["packages"][p] = "NOT_INSTALLED"
    try:
        import torch
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = torch.version.cuda
        info["cudnn_version"] = torch.backends.cudnn.version()
        info["gpu"] = [torch.cuda.get_device_name(i)
                       for i in range(torch.cuda.device_count())]
        info["gpu_total_memory_mib"] = [
            round(torch.cuda.get_device_properties(i).total_memory / 1048576)
            for i in range(torch.cuda.device_count())]
    except Exception as e:  # pragma: no cover
        info["cuda_error"] = str(e)
    try:  # NVIDIA 驱动版本（torch 不暴露，走 nvidia-smi）
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20)
        info["nvidia_driver"] = [x.strip() for x in out.stdout.splitlines() if x.strip()]
    except Exception as e:  # pragma: no cover
        info["nvidia_driver"] = f"UNAVAILABLE: {e}"
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--weights", default="weights/yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=MAIN_ARGS["epochs"])
    ap.add_argument("--batch", type=int, default=MAIN_ARGS["batch"])
    ap.add_argument("--imgsz", type=int, default=MAIN_ARGS["imgsz"])
    ap.add_argument("--device", default=MAIN_ARGS["device"])
    ap.add_argument("--seed", type=int, default=MAIN_ARGS["seed"])
    ap.add_argument("--cache", default="false")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    data_yaml = (ROOT / a.data).resolve()
    weights = (ROOT / a.weights).resolve()
    run_dir = RUNS / a.name

    if run_dir.exists() and any(run_dir.iterdir()) and not a.force:
        print(f"!! {run_dir} 已存在且非空，中止（加 --force 才继续）")
        return 1
    if not data_yaml.exists():
        print(f"!! 找不到 data yaml: {data_yaml}")
        return 1
    if not weights.exists():
        print(f"!! 找不到初始化权重: {weights}")
        return 1
    if "test" in data_yaml.name.lower():
        print("!! 拒绝在 test 配置上训练")
        return 1

    cache = False if str(a.cache).lower() in ("false", "none", "0", "off") else a.cache
    args = dict(MAIN_ARGS)
    args.update(epochs=a.epochs, batch=a.batch, imgsz=a.imgsz,
                device=a.device, seed=a.seed, cache=cache)

    t_start = datetime.now().astimezone()
    meta = {
        "run": a.name,
        "purpose": "split_v2 重训（Request 9）",
        "start_time": t_start.isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "main_args": {k: (v if not isinstance(v, bool) else v) for k, v in args.items()},
        "data_yaml": str(data_yaml.relative_to(ROOT)).replace("\\", "/"),
        "data_yaml_sha256": sha256_file(data_yaml),
        "data_yaml_content": data_yaml.read_text(encoding="utf-8"),
        "weights_init": str(weights.relative_to(ROOT)).replace("\\", "/"),
        "weights_init_sha256": sha256_file(weights),
        "split_v2_manifest_sha256": sha256_file(FLZ / "split_v2_manifest.csv"),
        "split_v2_freeze_manifest_sha256": sha256_file(FLZ / "split_v2_freeze_manifest.csv"),
        "code_commit": sh("git", "rev-parse", "HEAD"),
        "git_dirty_entries": len([l for l in sh("git", "status", "--porcelain").splitlines()]),
        "environment": env_snapshot(),
        "test_usage": "NOT USED — 训练全程只用 val；test 不在本脚本中出现",
        "eval_split_seen_during_training": ["train", "val"],
    }

    from ultralytics import YOLO
    model = YOLO(str(weights))
    meta["model_object"] = f"YOLO({weights.name}) 官方预训练权重"

    # --- 第八节要求在**训练前**就把配置落盘 ---
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_meta_prerun.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[V2] 训练前配置已保存 -> {run_dir / 'run_meta_prerun.json'}")

    t0 = time.time()
    model.train(data=str(data_yaml), project=str(RUNS), name=a.name,
                exist_ok=True, **args)
    meta["end_time"] = datetime.now().astimezone().isoformat(timespec="seconds")
    meta["wall_seconds"] = round(time.time() - t0, 1)
    meta["wall_hms"] = str(__import__("datetime").timedelta(seconds=int(time.time() - t0)))

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[V2] {a.name} 完成，用时 {meta['wall_hms']}")
    print(f"[V2] meta -> {run_dir / 'run_meta.json'}")
    print(f"[V2] best -> {run_dir / 'weights' / 'best.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
